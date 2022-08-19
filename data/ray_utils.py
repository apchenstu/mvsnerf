import torch
import numpy as np
from renderer import ndc2dist
from torch import searchsorted
from kornia import create_meshgrid

from renderer import depth2dist
from utils import index_point_feature


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal, (j - cent[1]) / focal, torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf
    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18
    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate
    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def dda(rays_o, rays_d, bbox_3D):
    inv_ray_d = 1.0 / (rays_d + 1e-6)
    t_min = (bbox_3D[:1] - rays_o) * inv_ray_d  # N_rays 3
    t_max = (bbox_3D[1:] - rays_o) * inv_ray_d
    t = torch.stack((t_min, t_max))  # 2 N_rays 3
    t_min = torch.max(torch.min(t, dim=0)[0], dim=-1, keepdim=True)[0]
    t_max = torch.min(torch.max(t, dim=0)[0], dim=-1, keepdim=True)[0]
    return t_min, t_max


def ray_marcher(rays,
                N_samples=64,
                lindisp=False,
                perturb=0,
                bbox_3D=None):
    """
    sample points along the rays
    Inputs:
        rays: ()

    Returns:

    """

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    if bbox_3D is not None:
        # cal aabb boundles
        near, far = dda(rays_o, rays_d, bbox_3D)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not lindisp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # rays_o.unsqueeze(1) (N_rays, 1, 3)
    # z_vals.unsqueeze(2) (N_rays, N_samples, 1)
    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    return xyz_coarse_sampled, rays_o, rays_d, z_vals


def ray_marcher_fine(rays,
                     density_volume,
                     z_vals,
                     pts_NDC,
                     N_importance=64,
                     lindisp=False):
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)

    pts_NDC = pts_NDC * 2 - 1.0
    sigma = index_point_feature(density_volume[None, None], pts_NDC)

    alpha = 1. - torch.exp(-torch.relu(sigma))  # *5e-3
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]

    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
    z_samples = sample_pdf(z_vals_mid, weights[:, 1:-1], N_importance).detach()

    z_vals = torch.sort(torch.cat([z_samples, z_vals], -1), -1)[0]

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    return xyz_coarse_sampled, rays_o, rays_d, z_vals


def get_rays_by_coord_np(H, W, focal, c2w, coords):
    i, j = (coords[:, 0] - W * 0.5) / focal, -(coords[:, 1] - H * 0.5) / focal
    dirs = np.stack([i, j, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d
