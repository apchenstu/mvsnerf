#!/usr/bin/env python
# coding: utf-8

# # import packages

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

torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# # Rendering video from finetuned ckpts

# ### functions

# In[48]:


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
    print('center in poses_avg', center)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)

    return c2w


# TODO: add
# def render_path_imgs(c2ws_all, focal):
#     T = c2ws_all[..., 3]
#
#     return render_poses


# TODO: change
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        # for theta in np.linspace(0., 5, N+1)[:-1]:
        # spiral
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)

        # 关于使用平均姿态的相关问题
        # 从训练集推算出来的平均姿态方向基本平行于z轴，因为训练集中大多数图片是正面的，
        # 但是存在一个问题，将z和平均姿态相乘后得到的方向也基本上和z平行，所以无论怎么调整看起来都是平行的，
        # 别用平均姿态看其他位置的照片，直接用世界坐标系即可！！！！
        # 但是需要用别的姿态大致估计一下位置参数
        c = np.array([(np.cos(theta) * theta) / 10, (-np.sin(theta) * theta) / 10, -0.1])

        # 这个是因为作者在读取并规范化相机姿态的时候作了poses*blender2opencv，转换了坐标系，
        # 我用的数据无需转换，但是这里加个负号就解决了，目前不影响什么，记住就行
        z = -(normalize(c - np.array([0, 0, -focal])))
        print("c", c)
        print("z", z)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_far, rads_scale=0.5, N_views=120):
    # center pose
    c2w = poses_avg(c2ws_all)
    print('poses_avg', c2w)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    print('near and far bounds', close_depth, inf_depth)
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz
    print(focal)

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:, :3, 3] - c2w[:3, 3][None]
    rads = np.percentile(np.abs(tt), 70, 0) * rads_scale
    print("rads", rads)
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)


def position2angle(position, N_views=16, N_rots=2):
    ''' nx3 '''
    position = normalize(position)
    theta = np.arccos(position[:, 2]) / np.pi * 180
    phi = np.arctan2(position[:, 1], position[:, 0]) / np.pi * 180
    return [theta, phi]


def pose_spherical_nerf(euler, radius=0.01):
    c2ws_render = np.eye(4)
    c2ws_render[:3, :3] = R.from_euler('xyz', euler, degrees=True).as_matrix()
    # 保留旋转矩阵的最后一列再乘个系数就能当作位置？
    c2ws_render[:3, 3] = c2ws_render[:3, :3] @ np.array([0.0, 0.0, -radius])
    return c2ws_render

# TODO: add
# def create_spheric_poses(radius, n_poses=120):
#     """
#     Create circular poses around z axis.
#     Inputs:
#         radius: the (negative) height and the radius of the circle.
#     Outputs:
#         spheric_poses: (n_poses, 3, 4) the poses in the circular path
#     """
#
#     def spheric_pose(theta, phi, radius):
#         trans_t = lambda t: np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, -0.9 * t],
#             [0, 0, 1, t],
#             [0, 0, 0, 1],
#         ])
#
#         rot_phi = lambda phi: np.array([
#             [1, 0, 0, 0],
#             [0, np.cos(phi), -np.sin(phi), 0],
#             [0, np.sin(phi), np.cos(phi), 0],
#             [0, 0, 0, 1],
#         ])
#
#         rot_theta = lambda th: np.array([
#             [np.cos(th), 0, -np.sin(th), 0],
#             [0, 1, 0, 0],
#             [np.sin(th), 0, np.cos(th), 0],
#             [0, 0, 0, 1],
#         ])
#
#         c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
#         c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
#         return c2w[:3]
#
#     spheric_poses = []
#     for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
#         spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
#     return np.stack(spheric_poses, 0)

# TODo: change
def nerf_video_path(c2ws, theta_range=10, phi_range=20, N_views=120):
    c2ws = torch.tensor(c2ws)
    mean_position = torch.mean(c2ws[:, :3, 3], dim=0).reshape(1, 3).cpu().numpy()
    rotvec = []
    for i in range(c2ws.shape[0]):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0]) > 180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
    # 采用欧拉角做平均的方法求旋转矩阵的平均
    rotvec = np.mean(np.stack(rotvec), axis=0)
    #     render_poses = [pose_spherical_nerf(rotvec)]
    render_poses = [pose_spherical_nerf(rotvec + np.array([angle, 0.0, -phi_range])) for angle in
                    np.linspace(-theta_range, theta_range, N_views // 4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec + np.array([theta_range, 0.0, angle])) for angle in
                     np.linspace(-phi_range, phi_range, N_views // 4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec + np.array([angle, 0.0, phi_range])) for angle in
                     np.linspace(theta_range, -theta_range, N_views // 4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec + np.array([-theta_range, 0.0, angle])) for angle in
                     np.linspace(phi_range, -phi_range, N_views // 4, endpoint=False)]
    # render_poses = torch.from_numpy(np.stack(render_poses)).float().to(device)
    return render_poses


# ### LLFF video rendering

# In[50]:


for i_scene, scene in enumerate(
        ['illusion']):  # 'horns','flower','orchids', 'room','leaves','fern','trex','fortress'
    # add --use_color_volume if the ckpts are fintuned with this flag
    cmd = f'--datadir /home/yuchen/mvsnerf/nerf_llff_data/{scene} ' \
          f'--dataset_name llff --imgScale_test {1.0}  ' \
          f'--netwidth 128 --net_type v0 '
          # f'--with_rgb_loss --batch_size {1024} ' \
          # f'--num_epochs {1} --pad {24} --N_vis {1} ' \

    # f'--netwidth 128 --net_type v0 '# Please check whether finetuning setting is same with rendering setting,
    # especially on use_color_volume, pad, and use_disp
    is_finetued = True  # set False if rendering without finetuning
    if is_finetued:
        cmd += f' --ckpt ./runs_fine_tuning/{scene}/ckpts/latest.tar'
    else:
        cmd += ' --ckpt ./ckpts/mvsnerf-v0.tar'

    args = config_parser(cmd.split())
    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim = 8 + 3 * 4
    # args.use_color_volume = False if not is_finetued else args.use_color_volume

    # create models
    if i_scene == 0 or is_finetued:
        render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True,
                                                                                    dir_embedder=False,
                                                                                    pts_embedder=True)
        filter_keys(render_kwargs_train)

        MVSNet = render_kwargs_train['network_mvs']
        render_kwargs_train.pop('network_mvs')

    datadir = args.datadir
    datatype = 'val'
    pad = 24  # the padding value should be same as your finetuning ckpt
    args.chunk = 5120

    dataset = dataset_dict[args.dataset_name](args, split=datatype)
    val_idx = dataset.img_idx

    save_dir = './results/videos'
    os.makedirs(save_dir, exist_ok=True)
    MVSNet.train()
    MVSNet = MVSNet.cuda()

    with torch.no_grad():

        c2ws_all = dataset.poses

        if is_finetued:
            # large baselien
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
            volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
            volume_feature = RefVolume(volume_feature.detach()).cuda()

            pad *= args.imgScale_test
            w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
            pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
            # 为啥这个地方只用训练集的相机姿态？
            # TODO: change
            c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale=10,
                                     N_views=30)  # you can enlarge the rads_scale if you want to render larger baseline
            # c2ws_render = nerf_video_path(c2ws_all[pair_idx], N_views=40)# you can enlarge the rads_scale if you want to render larger baseline
            # c2ws_render = create_spheric_poses(2, n_poses=10)
            # c2ws_render = render_path_imgs(c2ws_all[pair_idx])
        else:
            # neighboring views with position distance
            imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
            volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad, lindisp=args.use_disp)

            pad *= args.imgScale_test
            w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
            pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
            c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale=0.1,
                                     N_views=60)  # you can enlarge the rads_scale if you want to render larger baseline

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
            # print(depth_rays_preds)
            # img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)
            # TODO: change
            img_vis = rgb_rays * 255

            imageio.imwrite(f'{save_dir}/video_imgs_1/{str(i).zfill(4)}.png', img_vis.astype('uint8'))
            frames.append(img_vis.astype('uint8'))

    imageio.mimwrite(f'{save_dir}/ft_{scene}_spiral_test_v4.mp4', np.stack(frames), fps=10, quality=10)

# # render path generation

# In[25]:


render_poses = {}
datatype = 'val'
for i_scene, scene in enumerate(['illusion']):
    # add --use_color_volume if the ckpts are fintuned with this flag
    # cmd = f'--datadir /home/hengfei/Desktop/research/mvsnerf/xgaze/{scene}       --dataset_name llff --imgScale_test {1.0}     --ckpt ./runs_fine_tuning/{scene}-ft/ckpts/latest.tar'
    cmd = f'--datadir /home/yuchen/mvsnerf/nerf_llff_data/{scene} ' \
          f'--dataset_name llff --imgScale_test {1.0}  ' \
          f'--ckpt ./runs_fine_tuning/{scene}/ckpts/latest.tar'
          # f'--with_rgb_loss --batch_size {1024} ' \
          # f'--num_epochs {1} --pad {24} --N_vis {1} ' \

    args = config_parser(cmd.split())
    args.use_viewdirs = True

    print('============> rendering dataset <===================')
    dataset = dataset_dict[args.dataset_name](args, split=datatype)
    val_idx = dataset.img_idx

    imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)

    c2ws_all = dataset.poses
    w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
    pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
    c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale=0.1, N_views=60)

    # TODO: change
    render_poses[f'{scene}_c2ws'] = c2ws_render
    render_poses[f'{scene}_intrinsic'] = pose_source['intrinsics'][0].cpu().numpy()


torch.save(render_poses, './configs/video_path.th')
np.save('./configs/video_path.npy', render_poses)

# In[8]:

# TODO: change
rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    # num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None].repeat(num_cams, axis=0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


# In[57]:


import glob

# render_poses = {}
# datatype = 'val'
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
