import os, torch, cv2, re
import numpy as np


from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

from scipy.spatial.transform import Rotation as R

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
mse2psnr2 = lambda x : -10. * np.log(x) / np.log(10.)

def get_psnr(imgs_pred, imgs_gt):
    psnrs = []
    for (img,tar) in zip(imgs_pred,imgs_gt):
        psnrs.append(mse2psnr2(np.mean((img - tar.cpu().numpy())**2)))
    return np.array(psnrs)

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log



def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def abs_error_numpy(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    return np.abs(depth_pred - depth_gt)

def abs_error(depth_pred, depth_gt, mask):
    depth_pred, depth_gt = depth_pred[mask], depth_gt[mask]
    err = depth_pred - depth_gt
    return np.abs(err) if type(depth_pred) is np.ndarray else err.abs()

def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = abs_error(depth_pred, depth_gt, mask)
    acc_mask = errors < threshold
    return acc_mask.astype('float') if type(depth_pred) is np.ndarray else acc_mask.float()


# Ray helpers
def get_rays_mvs(H, W, intrinsic, c2w, N=1024, isRandom=True, is_precrop_iters=False, chunk=-1, idx=-1):

    device = c2w.device
    if isRandom:
        if is_precrop_iters and torch.rand((1,)) > 0.3:
            xs, ys = torch.randint(W//6, W-W//6, (N,)).float().to(device), torch.randint(H//6, H-H//6, (N,)).float().to(device)
        else:
            xs, ys = torch.randint(0,W,(N,)).float().to(device), torch.randint(0,H,(N,)).float().to(device)
    else:
        ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
        ys, xs = ys.reshape(-1), xs.reshape(-1)
        if chunk>0:
            ys, xs = ys[idx*chunk:(idx+1)*chunk], xs[idx*chunk:(idx+1)*chunk]
        ys, xs = ys.to(device), xs.to(device)

    dirs = torch.stack([(xs-intrinsic[0,2])/intrinsic[0,0], (ys-intrinsic[1,2])/intrinsic[1,1], torch.ones_like(xs)], -1) # use 1 instead of -1


    rays_d = dirs @ c2w[:3,:3].t() # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].clone()
    pixel_coordinates = torch.stack((ys,xs)) # row col
    return rays_o, rays_d, pixel_coordinates



def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2, far=6, pad=0, lindisp=False):
    '''
        point_samples [N_rays N_sample 3]
    '''

    N_rays, N_samples = point_samples.shape[:2]
    point_samples = point_samples.reshape(-1, 3)

    # wrap to ref view
    if w2c_ref is not None:
        R = w2c_ref[:3, :3]  # (3, 3)
        T = w2c_ref[:3, 3:]  # (3, 1)
        point_samples = torch.matmul(point_samples, R.t()) + T.reshape(1,3)

    if intrinsic_ref is not None:
        # using projection
        point_samples_pixel =  point_samples @ intrinsic_ref.t()
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  # normalize to 0~1
        if not lindisp:
            point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
        else:
            point_samples_pixel[:,2] = (1.0/point_samples_pixel[:,2]-1.0/near)/(1.0/far - 1.0/near)
    else:
        # using bounding box
        near, far = near.view(1,3), far.view(1,3)
        point_samples_pixel = (point_samples - near) / (far - near)  # normalize to 0~1
    del point_samples

    if pad>0:
        W_feat, H_feat = (inv_scale+1)/4.0
        point_samples_pixel[:,1] = point_samples_pixel[:,1] * H_feat / (H_feat + pad * 2) + pad / (H_feat + pad * 2)
        point_samples_pixel[:,0] = point_samples_pixel[:,0] * W_feat / (W_feat + pad * 2) + pad / (W_feat + pad * 2)

    point_samples_pixel = point_samples_pixel.view(N_rays, N_samples, 3)
    return point_samples_pixel

def build_rays(imgs, depths, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=0, is_precrop_iters=False, ref_idx=0, importanceSampling=False, with_depth=False, is_volume=False):
    '''

    Args:
        imgs: [N V C H W]
        depths: [N V H W]
        poses: w2c c2w intrinsic [N V 4 4] [B V levels 3 3)]
        init_depth_min: [B D H W]
        depth_interval:
        N_rays: int
        N_samples: same as D int
        level: int 0 == smalest
        near_fars: [B D 2]

    Returns:
        [3 N_rays N_samples]
    '''

    device = imgs.device

    N, V, C, H, W = imgs.shape
    w2c_ref, intrinsic_ref = pose_ref['w2cs'][ref_idx], pose_ref['intrinsics'][ref_idx]  # assume camera 0 is reference
    inv_scale = torch.tensor([W-1, H-1]).to(device)

    ray_coordinate_ref = []
    near_ref, far_ref = pose_ref['near_fars'][ref_idx, 0], pose_ref['near_fars'][ref_idx, 1]
    ray_coordinate_world, ray_dir_world, colors, depth_candidates = [],[],[],[]
    rays_os, rays_ds, cos_angles, rays_depths = [],[],[],[]

    for i in range(V-1,V):
        intrinsic = intrinsics[i]  #!!!!!! assuming batch size equal to 1
        c2w, w2c = c2ws[i].clone(), w2cs[i].clone()

        rays_o, rays_d, pixel_coordinates = get_rays_mvs(H, W, intrinsic, c2w, N_rays, is_precrop_iters=is_precrop_iters)   # [N_rays 3]


        # direction
        ray_dir_world.append(rays_d)    # toward camera [N_rays 3]

        # position
        rays_o = rays_o.reshape(1, 3)
        rays_o = rays_o.expand(N_rays, -1)
        rays_os.append(rays_o)

        # colors
        pixel_coordinates_int = pixel_coordinates.long()
        color = imgs[0, i, :, pixel_coordinates_int[0], pixel_coordinates_int[1]] # [3 N_rays]
        colors.append(color)

        if depths.shape[2] != 1:
            rays_depth = depths[0,i,pixel_coordinates_int[0], pixel_coordinates_int[1]]
            rays_depths.append(rays_depth)

        # travel along the rays
        if with_depth:
            depth_candidate = near_fars[pixel_coordinates_int[0], pixel_coordinates_int[1]].reshape(-1,1) #  [ray_samples N_samples]
        else:
            if importanceSampling:
                near, far = rays_depth - 0.1, rays_depth + 0.1
                near, far = near.view(N_rays, 1), far.view(N_rays, 1)
            else:
                near, far = near_fars[0, i, 0], near_fars[0, i, 1]

            t_vals = torch.linspace(0., 1., steps=N_samples).view(1,N_samples).to(device)
            depth_candidate = near * (1. - t_vals) + far * (t_vals)
            depth_candidate = depth_candidate.expand([N_rays, N_samples])

            # get intervals between samples
            mids = .5 * (depth_candidate[..., 1:] + depth_candidate[..., :-1])
            upper = torch.cat([mids, depth_candidate[..., -1:]], -1)
            lower = torch.cat([depth_candidate[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(depth_candidate.shape, device=device)
            depth_candidate = lower + (upper - lower) * t_rand

        point_samples = rays_o.unsqueeze(1) + depth_candidate.unsqueeze(-1) * rays_d.unsqueeze(1)   #  [ray_samples N_samples 3 ]
        depth_candidates.append(depth_candidate) #  [ray_samples N_rays]

        # position
        ray_coordinate_world.append(point_samples)  # [ray_samples N_samples 3] xyz in [0,1]
        points_ndc = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=near_ref, far=far_ref, pad=pad)

        ray_coordinate_ref.append(points_ndc)

    ndc_parameters = {'w2c_ref':w2c_ref, 'intrinsic_ref':intrinsic_ref, 'inv_scale':inv_scale, 'near':near_ref, 'far':far_ref}
    colors = torch.cat(colors, dim=1).permute(1,0)
    rays_depths = torch.cat(rays_depths) if len(rays_depths)>0 else None
    depth_candidates = torch.cat(depth_candidates, dim=0)
    ray_dir_world = torch.cat(ray_dir_world, dim=0)
    ray_coordinate_world = torch.cat(ray_coordinate_world, dim=0)
    rays_os = torch.cat(rays_os, dim=0).permute(1,0)
    ray_coordinate_ref = torch.cat(ray_coordinate_ref, dim=0)

    return ray_coordinate_world, ray_dir_world, colors, ray_coordinate_ref, depth_candidates, rays_os, rays_depths, ndc_parameters

def build_rays_test(H,W, tgt_to_world, world_to_ref, intrinsic, near_fars_ref, near_fars, N_samples, pad=0, ref_idx=0, use_cpu=False, chunk=-1, idx=-1):
    '''

    Args:
        extrinsic intrinsic [4 4] [3 3)]
        N_samples: same as D int
        depth_values: [B D]

    Returns:
        [3 N_rays N_samples]
    '''

    device = torch.device("cpu") if use_cpu else tgt_to_world.device
    if use_cpu:
        tgt_to_world,world_to_ref = tgt_to_world.clone().to(device),world_to_ref.clone().to(device)
        intrinsic, near_fars_ref, near_fars = intrinsic.clone().to(device),near_fars_ref.clone().to(device),near_fars.clone().to(device)
    inv_scale = torch.tensor([W - 1, H - 1]).to(device)


    ray_coordinate_world, ray_dir_world, colors, depth_candidates = [],[],[],[]
    rays_os, rays_ds = [],[]

    intrinsic_render =  intrinsic if intrinsic.dim()==2 else intrinsic.mean(0)
    rays_o, rays_d, pixel_coordinates = get_rays_mvs(H, W, intrinsic_render, tgt_to_world, isRandom=False, chunk=chunk, idx=idx)
    ray_samples = H * W if chunk < 0 else pixel_coordinates.shape[-1]


    # direction
    ray_dir_world.append(rays_d)    # toward camera [N_rays 3]

    # position
    rays_o = rays_o.reshape(1,3)
    rays_o = rays_o.expand(ray_samples, -1)
    rays_os.append(rays_o)

    # travel along the rays
    near, far = near_fars[0], near_fars[1]
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    depth_candidate = near * (1. - t_vals) + far * (t_vals)
    depth_candidate = depth_candidate.expand([ray_samples, N_samples])
    point_samples = rays_o.unsqueeze(1) + depth_candidate.unsqueeze(-1) * rays_d.unsqueeze(1)    #  [3 ray_samples N_samples]
    depth_candidates.append(depth_candidate) #  [ray_samples N_rays]

    # position
    near, far = near_fars_ref[ref_idx, 0], near_fars_ref[ref_idx, 1]
    ray_coordinate_world.append(point_samples)  # [ray_samples N_samples 3] xyz in [0,1]
    ray_coordinate_ref = get_ndc_coordinate(world_to_ref, intrinsic, point_samples, inv_scale, near=near, far=far, pad=pad)

    ndc_parameters = {'w2c_ref': world_to_ref, 'intrinsic_ref': intrinsic, 'inv_scale': inv_scale, 'near': near, 'far': far}
    depth_candidates = torch.cat(depth_candidates, dim=0)
    ray_dir_world = torch.cat(ray_dir_world, dim=0)
    ray_coordinate_world = torch.cat(ray_coordinate_world, dim=0)
    rays_os = torch.cat(rays_os, dim=0)

    return ray_coordinate_world, ray_dir_world,ray_coordinate_ref, depth_candidates, rays_os, ndc_parameters


def build_color_volume(point_samples, pose_ref, imgs, img_feat=None, downscale=1.0, with_mask=False):
    '''
    point_world: [N_ray N_sample 3]
    imgs: [N V 3 H W]
    '''

    device = imgs.device
    N, V, C, H, W = imgs.shape
    inv_scale = torch.tensor([W - 1, H - 1]).to(device)

    C += with_mask
    C += 0 if img_feat is None else img_feat.shape[2]
    colors = torch.empty((*point_samples.shape[:2], V*C), device=imgs.device, dtype=torch.float)
    for i,idx in enumerate(range(V)):

        w2c_ref, intrinsic_ref = pose_ref['w2cs'][idx], pose_ref['intrinsics'][idx].clone()  # assume camera 0 is reference
        point_samples_pixel = get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale)[None]
        grid = point_samples_pixel[...,:2]*2.0-1.0

        # img = F.interpolate(imgs[:, idx], scale_factor=downscale, align_corners=True, mode='bilinear',recompute_scale_factor=True) if downscale != 1.0 else imgs[:, idx]
        data = F.grid_sample(imgs[:, idx], grid, align_corners=True, mode='bilinear', padding_mode='border')
        if img_feat is not None:
            data = torch.cat((data,F.grid_sample(img_feat[:,idx], grid, align_corners=True, mode='bilinear', padding_mode='zeros')),dim=1)

        if with_mask:
            in_mask = ((grid >-1.0)*(grid < 1.0))
            in_mask = (in_mask[...,0]*in_mask[...,1]).float()
            data = torch.cat((data,in_mask.unsqueeze(1)), dim=1)

        colors[...,i*C:i*C+C] = data[0].permute(1, 2, 0)
        del grid, point_samples_pixel, data

    return colors


def normal_vect(vect, dim=-1):
    return vect / (torch.sqrt(torch.sum(vect**2,dim=dim,keepdim=True))+1e-7)

def get_ptsvolume(H, W, D, pad, near_far, intrinsic, c2w):
    device = intrinsic.device
    near,far = near_far

    corners = torch.tensor([[-pad,-pad,1.0],[W+pad,-pad,1.0],[-pad,H+pad,1.0],[W+pad,H+pad,1.0]]).float().to(intrinsic.device)
    corners = torch.matmul(corners, torch.inverse(intrinsic).t())

    linspace_x = torch.linspace(corners[0, 0], corners[1, 0], W+2*pad)
    linspace_y = torch.linspace(corners[ 0, 1], corners[2, 1], H+2*pad)
    ys, xs = torch.meshgrid(linspace_y, linspace_x)  # HW
    near_plane = torch.stack((xs,ys,torch.ones_like(xs)),dim=-1).to(device)*near
    far_plane = torch.stack((xs,ys,torch.ones_like(xs)),dim=-1).to(device)*far

    linspace_z = torch.linspace(1.0, 0.0, D).view(D,1,1,1).to(device)
    pts = linspace_z*near_plane + (1.0-linspace_z)*far_plane
    pts = torch.matmul(pts.view(-1,3), c2w[:3,:3].t()) + c2w[:3,3].view(1,3)

    return pts.view(D*(H+pad*2),W+pad*2,3)

def index_point_feature(volume_feature, ray_coordinate_ref, chunk=-1):
        ''''
        Args:
            volume_color_feature: [B, G, D, h, w]
            volume_density_feature: [B C D H W]
            ray_dir_world:[3 ray_samples N_samples]
            ray_coordinate_ref:  [3 N_rays N_samples]
            ray_dir_ref:  [3 N_rays]
            depth_candidates: [N_rays, N_samples]
        Returns:
            [N_rays, N_samples]
        '''

        device = volume_feature.device
        H, W = ray_coordinate_ref.shape[-3:-1]


        if chunk != -1:
            features = torch.zeros((volume_feature.shape[1],H,W), device=volume_feature.device, dtype=torch.float, requires_grad=volume_feature.requires_grad)
            grid = ray_coordinate_ref.view(1, 1, 1, H * W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
            for i in range(0, H*W, chunk):
                features[:,i:i + chunk] = F.grid_sample(volume_feature, grid[:,:,:,i:i + chunk], align_corners=True, mode='bilinear')[0]
            features = features.permute(1,2,0)
        else:
            grid = ray_coordinate_ref.view(-1, 1, H,  W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
            features = F.grid_sample(volume_feature, grid, align_corners=True, mode='bilinear')[:,:,0].permute(2,3,0,1).squeeze()#, padding_mode="border"
        return features





def to_tensor_cuda(data, device, filter):
    for item in data.keys():

        if item in filter:
            continue

        if type(data[item]) is np.ndarray:
            data[item] = torch.tensor(data[item], dtype=torch.float32, device= device)
        else:
            data[item] = data[item].float().to(device)
    return data


def to_cuda(data, device, filter):
    for item in data.keys():
        if item in filter:
            continue

        data[item] = data[item].float().to(device)
    return data

def tensor_unsqueeze(data, filter):
    for item in data.keys():
        if item in filter:
            continue

        data[item] = data[item][None]
    return data

def filter_keys(dict):
    dict.pop('N_samples')
    if 'ndc' in dict.keys():
        dict.pop('ndc')
    if 'lindisp' in dict.keys():
        dict.pop('lindisp')
    return dict

def sub_selete_data(data_batch, device, idx, filtKey=[], filtIndex=['view_ids_all','c2ws_all','scan','bbox','w2ref','ref2w','light_id','ckpt','idx']):
    data_sub_selete = {}
    for item in data_batch.keys():
        data_sub_selete[item] = data_batch[item][:,idx].float() if (item not in filtIndex and torch.is_tensor(item) and item.dim()>2) else data_batch[item].float()
        if not data_sub_selete[item].is_cuda:
            data_sub_selete[item] = data_sub_selete[item].to(device)
    return data_sub_selete

def detach_data(dictionary):
    dictionary_new = {}
    for key in dictionary.keys():
        dictionary_new[key] = dictionary[key].detach().clone()
    return dictionary_new

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def gen_render_path_spherical(theta, phi, radius=1.0):
    blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w





from scipy.interpolate import CubicSpline
def gen_render_path_pixelNeRF(c2ws, N_views=30):
    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = np.array(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = N_views // 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([450.0, 450.0, 450.0, 450.0, 450.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat, bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        R = R.from_quat(new_q)
        t = R[:, 2] * scale
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t
        new_pose = c2ws[0,0] @ new_pose
        render_poses.append(new_pose)
    render_poses = torch.stack(render_poses, dim=0)
    return render_poses



#################################################  MVS  helper functions   #####################################
from kornia.utils import create_meshgrid

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory



        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid


###############################  render path  ####################################
def pose_spherical_nerf(euler, radius=4.0):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,-radius])
    return c2ws_render

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def pose_spherical_dtu(radii, focus_depth, n_poses=120, world_center=np.array([0,0,0])):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center+world_center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0) @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])  # (n_poses, 3, 4)

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from warmup_scheduler import GradualWarmupScheduler
def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)

    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)
    return scheduler


####  pairing  ####
def get_nearest_pose_ids(tar_pose, ref_poses, num_select):
    '''
    Args:
        tar_pose: target pose [N, 4, 4]
        ref_poses: reference poses [M, 4, 4]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''

    dists = np.linalg.norm(tar_pose[:,None,:3,3] - ref_poses[None,:,:3,3], axis=-1)

    sorted_ids = np.argsort(dists, axis=-1)
    selected_ids = sorted_ids[:,:num_select]
    return selected_ids
