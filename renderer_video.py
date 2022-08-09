#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os, imageio

root = '/home/yuchen/mvsnerf'
os.chdir(root)
sys.path.append(root)

from opt import config_parser
from data import dataset_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# models
from models import *
from renderer import *
from data.ray_utils import get_rays
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm

from skimage.metrics import structural_similarity

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers

from data.ray_utils import ray_marcher

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# # Rendering video from finetuned ckpts

# In[ ]:


def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs


def unpreprocess(data, shape=(1, 1, 3, 1, 1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_far, rads_scale=0.5, N_views=120):
    # center pose
    c2w = poses_avg(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:, :3, 3] - c2w[:3, 3][None]
    rads = np.percentile(np.abs(tt), 70, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


def position2angle(position, N_views=16, N_rots=2):
    ''' nx3 '''
    position = normalize(position)
    theta = np.arccos(position[:, 2]) / np.pi * 180
    phi = np.arctan2(position[:, 1], position[:, 0]) / np.pi * 180
    return [theta, phi]


def pose_spherical_nerf(euler, radius=4.0):
    c2ws_render = np.eye(4)
    c2ws_render[:3, :3] = R.from_euler('xyz', euler, degrees=True).as_matrix()
    c2ws_render[:3, 3] = c2ws_render[:3, :3] @ np.array([0.0, 0.0, -radius])
    return c2ws_render


def nerf_video_path(c2ws, theta_range=10, phi_range=20, N_views=120):
    mean_position = torch.mean(c2ws[:, :3, 3], dim=0).reshape(1, 3).cpu().numpy()
    rotvec = []
    for i in range(c2ws.shape[0]):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0]) > 180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
    rotvec = np.mean(np.stack(rotvec), axis=0)
    #     render_poses = [pose_spherical_nerf(rotvec)]
    render_poses = [pose_spherical_nerf(rotvec + np.array([angle, 0.0, -phi_range]), 4.0) for angle in
                    np.linspace(-theta_range, theta_range, N_views // 4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec + np.array([theta_range, 0.0, angle]), 4.0) for angle in
                     np.linspace(-phi_range, phi_range, N_views // 4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec + np.array([angle, 0.0, phi_range]), 4.0) for angle in
                     np.linspace(theta_range, -theta_range, N_views // 4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec + np.array([-theta_range, 0.0, angle]), 4.0) for angle in
                     np.linspace(phi_range, -phi_range, N_views // 4, endpoint=False)]
    render_poses = torch.from_numpy(np.stack(render_poses)).float().to(device)
    return render_poses


# In[ ]:


for i_scene, scene in enumerate(['Penrose']):  # 'horns','flower','orchids', 'room','leaves','fern','trex','fortress', '3D_Graffiti', 'illusion', 't-rex'
    # add --use_color_volume if the ckpts are fintuned with this flag
    cmd = f'--datadir /home/yuchen/mvsnerf/nerf_llff_data/{scene} ' \
          f'--dataset_name llff --imgScale_test {1.0}  ' \
          f'--netwidth 128 --net_type v0 '
    # f'--with_rgb_loss --batch_size {1024} ' \
    # f'--num_epochs {1} --pad {24} --N_vis {1} '\

    is_finetued = False  # set False if rendering without finetuning
    if is_finetued:
        cmd += f'--ckpt ./runs_fine_tuning/{scene}/ckpts/latest.tar'
        name = 'ft_'
    else:
        cmd += '--ckpt ./ckpts/mvsnerf-v0.tar'
        name = ''

    args = config_parser(cmd.split())
    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim = 8 + 3 * 4  #TODO: why
    #     args.use_color_volume = False if not is_finetued else args.use_color_volume

    # create models
    if i_scene == 0 or is_finetued:
        # Create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars = \
            create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(render_kwargs_train)
        # Create mvs model
        MVSNet = render_kwargs_train['network_mvs']
        render_kwargs_train.pop('network_mvs')

    datadir = args.datadir
    datatype = 'val'
    pad = 24  # the padding value should be same as your finetuning ckpt
    args.chunk = 5120

    dataset = dataset_dict[args.dataset_name](args, split=datatype)
    val_idx = dataset.img_idx

    save_dir = f'./results'
    os.makedirs(save_dir, exist_ok=True)
    MVSNet.train()
    MVSNet = MVSNet.cuda()

    with torch.no_grad():

        c2ws_all = dataset.poses

        if is_finetued:
            # large baseline
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
            volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
            volume_feature = RefVolume(volume_feature.detach()).cuda()

            pad *= args.imgScale_test
            w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
            pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
            c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale=0.6,
                                     N_views=180)  # you can enlarge the rads_scale if you want to render larger baseline
        else:
            # neighboring views with position distance
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
            volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad, lindisp=args.use_disp)

            pad *= args.imgScale_test
            w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
            pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
            c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale=0.6,
                                     N_views=180)  # you can enlarge the rads_scale if you want to render larger baseline

        c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)

        imgs_source = unpreprocess(imgs_source)

        try:
            tqdm._instances.clear()
        except Exception:
            pass

        frames = []
        img_directions = dataset.directions.to(device)
        for i, c2w in enumerate(tqdm(c2ws_render)):
            torch.cuda.empty_cache()

            rays_o, rays_d = get_rays(img_directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d,
                              near_far_source[0] * torch.ones_like(rays_o[:, :1]),
                              near_far_source[1] * torch.ones_like(rays_o[:, :1])],
                             1).to(device)  # (H*W, 3)

            N_rays_all = rays.shape[0]
            rgb_rays, depth_rays_preds = [], []
            for chunk_idx in range(N_rays_all // args.chunk + int(N_rays_all % args.chunk > 0)):
                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(
                    rays[chunk_idx * args.chunk:(chunk_idx + 1) * args.chunk],
                    N_samples=args.N_samples, lindisp=args.use_disp)

                # Converting world coordinate to ndc coordinate
                H, W = imgs_source.shape[-2:]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=near_far_source[0], far=near_far_source[1], pad=pad,
                                             lindisp=args.use_disp)

                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
                                                                      xyz_NDC, z_vals, rays_o, rays_d,
                                                                      volume_feature, imgs_source,
                                                                      **render_kwargs_train)

                rgb, depth_pred = torch.clamp(rgb.cpu(), 0, 1.0).numpy(), depth_pred.cpu().numpy()
                rgb_rays.append(rgb)
                depth_rays_preds.append(depth_pred)

            depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
            depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)

            rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
            H_crop, W_crop = np.array(rgb_rays.shape[:2]) // 20
            rgb_rays = rgb_rays[H_crop:-H_crop, W_crop:-W_crop]
            depth_rays_preds = depth_rays_preds[H_crop:-H_crop, W_crop:-W_crop]
            img_vis = np.concatenate((rgb_rays * 255, depth_rays_preds), axis=1)

            # imageio.imwrite(f'{save_dir}/video_imgs_1/{str(i).zfill(4)}.png', img_vis.astype('uint8'))
            frames.append(img_vis.astype('uint8'))

    imageio.mimwrite(f'{save_dir}/{name}{scene}.mov', np.stack(frames), fps=30, quality=10)
    os.system(f"/home/yuchen/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg -i {save_dir}/{name}{scene}.mov -vcodec h264 -acodec mp2 {save_dir}/{name}{scene}.mp4")
    os.system(f"rm {save_dir}/{name}{scene}.mov")
# # render path generation

# In[ ]:


# render_poses = {}
# datatype = 'val'
# for i_scene, scene in enumerate(['illusion']):
#     # add --use_color_volume if the ckpts are fintuned with this flag
#     cmd = f'--datadir /home/yuchen/mvsnerf/nerf_llff_data/{scene} ' \
#           f'--dataset_name llff --imgScale_test {1.0}  ' \
#           f'--ckpt ./runs_fine_tuning/{scene}/ckpts/latest.tar'
#           # f'--with_rgb_loss --batch_size {1024} ' \
#           # f'--num_epochs {1} --pad {24} --N_vis {1} ' \
#
#     args = config_parser(cmd.split())
#     args.use_viewdirs = True
#
#
#     print('============> rendering dataset <===================')
#     dataset = dataset_dict[args.dataset_name](args, split=datatype)
#     val_idx = dataset.img_idx
#
#
#     imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
#
#     c2ws_all = dataset.poses
#     w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
#     pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
#     c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale = 0.5, N_views=60)
#
#     render_poses[f'{scene}_near_far_source'] = near_far_source
#     render_poses[f'{scene}_c2ws_no_ft'] = c2ws_render
#     render_poses[f'{scene}_intrinsic_no_ft'] = pose_source['intrinsics'][0].cpu().numpy()
#
#
# torch.save(render_poses, './configs/video_path.th')
# np.save('./configs/video_path.npy',render_poses)

