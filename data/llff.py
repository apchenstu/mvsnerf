import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg



def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    # 草，这个地方源代码没有乘这个blender2opencv，做这个操作相当于把相机转换到另一个坐标系了，和一般的nerf坐标系不同
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    print('center in center_poses',poses_centered[:, :3, 3].mean(0))

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv


def create_spiral_poses(radii, focus_depth, n_poses=120):
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

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)z


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, args, split='train', spheric_poses=True, load_ref=False):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        downsample = args.imgScale_train if split == 'train' else args.imgScale_test
        self.img_wh = (int(960*downsample),int(640*downsample))
        assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
            'image width must be divisible by 32, you may need to modify the imgScale'
        self.spheric_poses = spheric_poses
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # self.blender2opencv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        if not load_ref:
            self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        # load full resolution image then resize
        if self.split in ['train', 'val']:
            print(len(poses_bounds) , len(self.image_paths),self.root_dir)
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.focal = [self.focal* self.img_wh[0] / W, self.focal* self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)
        # print('pose_avg in read_meta', self.pose_avg)
        # self.poses = poses @ self.blender2opencv

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        print('scale_factor', scale_factor)
        # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # sub select training views from pairing file
        if os.path.exists('configs/pairs.th'):
            name = os.path.basename(self.root_dir)
            self.img_idx = torch.load('configs/pairs.th')[f'{name}_{self.split}']

            print(f'===> {self.split}ing index: {self.img_idx}')

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)


        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        for i in self.img_idx:

            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)


            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
                # near plane is always at 1.0
                # near and far in NDC are always 0 and 1
                # See https://github.com/bmild/nerf/issues/34
            else:
                # near = self.bounds.min()
                # far = min(8 * near, self.bounds.max())  # focus on central object only
                near = self.bounds[i][0]*0.8
                far = self.bounds[i][1]*1.2  # focus on central object only

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         near * torch.ones_like(rays_o[:, :1]),
                                         far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)

        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
        elif 'val' == self.split:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

    def read_source_views(self, pair_idx=None, device=torch.device("cpu")):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        # load full resolution image then resize
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
        print('original focal', focal)

        focal = [focal* self.img_wh[0] / W, focal* self.img_wh[1] / H]
        print('porcessed focal', focal)

        # Step 2: correct poses
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        poses, _ = center_poses(poses, self.blender2opencv)
        # poses = poses @ self.blender2opencv

        # sub select training views from pairing file
        if pair_idx is None:
            name = os.path.basename(self.root_dir)
            pair_idx = torch.load('configs/pairs.th')[f'{name}_train'][:3]

            # positions = poses[pair_idx,:3,3]
            # dis = np.sum(np.abs(positions - np.mean(positions, axis=0, keepdims=True)), axis=-1)
            # pair_idx = [pair_idx[i] for i in np.argsort(dis)[:3]]
            print(f'====> ref idx: {pair_idx}')


        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., 3] /= scale_factor

        src_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])

        w, h = self.img_wh

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i, idx in enumerate(pair_idx):
            c2w = torch.eye(4).float()
            image_path = image_paths[idx]
            c2w[:3] = torch.FloatTensor(poses[idx])
            w2c = torch.inverse(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = torch.eye(4)
            intrinsic = torch.tensor([[focal[0], 0, w / 2], [0, focal[1], h / 2], [0, 0, 1]]).float()
            intrinsics.append(intrinsic.clone())
            intrinsic[:2] = intrinsic[:2] / 4   # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = torch.inverse(proj_mat_l)
                proj_mats += [torch.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]


            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.stack(c2ws).float().to(device)
        pose_source['w2cs'] = torch.stack(w2cs).float().to(device)
        pose_source['intrinsics'] = torch.stack(intrinsics).float().to(device)


        near_far_source = [bounds[pair_idx].min()*0.8,bounds[pair_idx].max()*1.2]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
            # sample = {'rays': self.all_rays[view,ray_idx],
            #           'rgbs': self.all_rgbs[view,ray_idx]}
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
            # sample['idx'] = view
        else:
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]

            sample = {'rays': rays,
                      'rgbs': img}

        return sample
