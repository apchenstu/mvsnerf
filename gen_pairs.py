import json,torch
import sys,os
import numpy as np
root = '/home/yuchen/mvsnerf'
os.chdir(root)
sys.path.append(root)
pairs = torch.load('./configs/pairs.th')

# llff
root_dir = './nerf_llff_data/'
for scene in ['triangle_5views']:  # t-rex, hexagonal, 3D_Graffiti, Penrose
    poses_bounds = np.load(os.path.join(root_dir, scene, 'poses_bounds.npy'))  # (N_images, 11) -> (121, 17)
    N_images = poses_bounds.shape[0]
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    poses = np.concatenate([poses[..., 1:2], - poses[..., :1], poses[..., 2:4]], -1)

    ref_position = np.mean(poses[..., 3], axis=0, keepdims=True)
    dist = np.sum(np.abs(poses[..., 3] - ref_position), axis=-1)
    # pair_idx = np.argsort(dist)[:121]
    pair_idx = torch.randperm(len(poses))[:N_images].tolist()

    pairs[f'{scene}_test'] = pair_idx[::6]
    pairs[f'{scene}_val'] = pair_idx[::6]
    pairs[f'{scene}_train'] = np.delete(pair_idx, range(0, N_images, 6))

    # pairs[f'{scene}_test'] = pair_idx[1]
    # pairs[f'{scene}_val'] = pair_idx[1]
    # pairs[f'{scene}_train'] = np.delete(pair_idx, 1)
torch.save(pairs, './configs/pairs.th')