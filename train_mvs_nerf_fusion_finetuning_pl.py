from opt import config_parser
from torch.utils.data import DataLoader

from data import dataset_dict

# models
from models import *
from renderer import *
from utils import *
from data.ray_utils import ray_marcher, get_ray_directions, get_rays

from tqdm import tqdm
import imageio


# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss


def update_volume(canonical_volume, canonical_alpha, canonical_weights, ray_feat, ray_ndc_pts,
                  ray_alpha, ray_weight):
    '''
        canonical_volume, canonical_density, canonical_weightsl: [1,C,D,H,W]
        ray_feat, ray_ndc_pts, ray_weight: [N_ray, N_sample, C]
    '''
    device = canonical_volume.device
    WHD = canonical_volume.shape[-3:][::-1]
    voxel_size = 1.0 / (torch.tensor(WHD).to(device) - 1)


    N_points = ray_ndc_pts.shape[0] * ray_ndc_pts.shape[1]
    ray_alpha = ray_alpha.view(N_points, -1)
    ray_feat, ray_ndc_pts, ray_weight = ray_feat.view(N_points, -1), ray_ndc_pts.view(N_points, -1), ray_weight.view(
        N_points, -1)

    # local index
    vox_idx = ray_ndc_pts / voxel_size.view(1, 3)
    local_coordinate = vox_idx - torch.floor(vox_idx)
    vox_idx = vox_idx.long()

    # filter voxel outside the volume
    W, H, D = WHD
    mask = (vox_idx[:, 0] >= 0) * (vox_idx[:, 1] >= 0) * (vox_idx[:, 2] >= 0) * (vox_idx[:, 0] < W - 1) * (
                vox_idx[:, 1] < H - 1) * (vox_idx[:, 2] < D - 1)
    if torch.sum(mask) == 0:
        return

    ray_alpha, vox_idx, local_coordinate = ray_alpha[mask], vox_idx[mask], local_coordinate[mask]
    ray_feat, ray_ndc_pts, ray_weight = ray_feat[mask], ray_ndc_pts[mask], ray_weight[mask]


    # bicube intepolation
    for shiftment in [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]:
        x, y, z = shiftment
        weight_local = torch.abs(
            local_coordinate - torch.tensor([x, y, z], device=local_coordinate.device).float().view(1, 3))
        weight_local = (weight_local[:, :1] * weight_local[:, 1:2] * weight_local[:, 2:]).t()

        canonical_weights[0, :1, vox_idx[:, 2] + x, vox_idx[:, 1] + y, vox_idx[:, 0] + z] += weight_local
        canonical_volume[0, :, vox_idx[:, 2] + x, vox_idx[:, 1] + y, vox_idx[:, 0] + z] += weight_local * ray_feat.t()
        canonical_alpha[0, :1, vox_idx[:, 2] + x, vox_idx[:, 1] + y, vox_idx[:, 0] + z] += weight_local * ray_alpha.t()

class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8+12
        self.args.dir_dim = 3
        self.idx = 0

        self.loss = SL1Loss()

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']
        self.render_kwargs_train.pop('network_mvs')

        dataset = dataset_dict[self.args.dataset_name]
        self.train_dataset = dataset(args, split='train')
        self.val_dataset   = dataset(args, split='val')

        args.use_color_volume = False
        self.volume_dim = [128,128,128]
        self.near_far_source = self.train_dataset.near_far
        self.bbox_3d = self.train_dataset.bbox_3d.to(device)
        self.fuse_local_volumes()
        self.grad_vars += list(self.volume.parameters())
        args.use_color_volume = True

        if args.N_importance:
            linspace_x = torch.linspace(0, 1.0, self.volume_dim[0])  # pixel shift to align the pixels
            linspace_y = torch.linspace(0, 1.0, self.volume_dim[1])
            linspace_z = torch.linspace(0, 1.0, self.volume_dim[2])
            zs, ys, xs = torch.meshgrid(linspace_z, linspace_y, linspace_x)  # DHW
            self.vox_pts = torch.stack((xs, ys, zs), -1).reshape(self.volume_dim[2] * self.volume_dim[1], self.volume_dim[0], 3).to(device)
            self.vox_pts = self.vox_pts*2 - 1.0
            del ys, xs, zs

    def fuse_local_volumes(self):

        feat_dim = 8+12
        volume_dim = self.volume_dim

        canonical_sigma = torch.zeros((1, 1, volume_dim[2], volume_dim[1], volume_dim[0])).to(device)
        canonical_weights = torch.zeros((1, 1, volume_dim[2], volume_dim[1], volume_dim[0])).to(device)
        canonical_volume = torch.zeros((1, feat_dim, volume_dim[2], volume_dim[1], volume_dim[0])).to(device)


        pairs = np.array(self.train_dataset.pair_idx[0])
        c2w_render = self.train_dataset.load_poses_all()[pairs]

        W,H = self.train_dataset.img_wh
        H, W = H // 4, W // 4
        img_directions = get_ray_directions(H, W, torch.tensor(self.train_dataset.focal) / 4.0).to(device)

        with torch.no_grad():
            for i, c2w in enumerate(tqdm(c2w_render)):
                torch.cuda.empty_cache()

                # find nearest image idx from training views
                positions = c2w_render[:, :3, 3]
                dis = np.sum(np.abs(positions - c2w[:3, 3:].T), axis=-1)
                pair_idx = pairs[np.argsort(dis)[:3]]

                imgs_source, proj_mats, near_far_source, pose_source = self.train_dataset.read_source_views(pair_idx=pair_idx,device=device)
                volume_feature, _, _ = self.MVSNet(imgs_source, proj_mats, near_far_source, pad=args.pad)
                imgs_source = self.unpreprocess(imgs_source)
                if 0 == i:
                    self.pose_source_ref = pose_source

                rays_o, rays_d = get_rays(img_directions, torch.from_numpy(c2w).float().to(device))  # both (h*w, 3)
                rays = torch.cat([rays_o, rays_d,
                                  near_far_source[0] * torch.ones_like(rays_o[:, :1]),
                                  near_far_source[1] * torch.ones_like(rays_o[:, :1])], 1).to(device)  # (H*W, 3)

                N_rays_all = rays.shape[0]
                rgb_rays, depth_rays_preds = [], []
                for chunk_idx in range(N_rays_all // args.chunk + int(N_rays_all % args.chunk > 0)):

                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(
                        rays[chunk_idx * args.chunk:(chunk_idx + 1) * args.chunk], N_samples=128)

                    # Converting world coordinate to ndc coordinate
                    inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                    w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                    intrinsic_ref[:2] *= 0.25

                    xyz_ndc = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                                 near=near_far_source[0], far=near_far_source[1],
                                                 pad=args.pad * 0.25)

                    # rendering
                    rgb, ray_feat, ray_weight, depth_pred, ray_sigma, _ = rendering(args, pose_source, xyz_coarse_sampled,
                                                                                    xyz_ndc, z_vals, rays_o, rays_d,
                                                                                    volume_feature, imgs_source,
                                                                                    **self.render_kwargs_train)

                    ray_ndc = (xyz_coarse_sampled - self.bbox_3d[0].view(1, 1, 3)) / (self.bbox_3d[1] - self.bbox_3d[0]).view(1, 1, 3)
                    update_volume(canonical_volume, canonical_sigma, canonical_weights, ray_feat, ray_ndc, ray_sigma, ray_weight)

                #     rgb, depth_pred = torch.clamp(rgb.cpu(), 0, 1.0).numpy(), depth_pred.cpu().numpy()
                #     rgb_rays.append(rgb)
                #     depth_rays_preds.append(depth_pred)
                #
                # depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
                # depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
                #
                # rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
                # img_vis = np.concatenate((rgb_rays * 255, depth_rays_preds), axis=1)
                # imageio.imwrite(f'/mnt/new_disk2/anpei/code/MVS-NeRF/results/test4/{i:03d}.png', img_vis.astype('uint8'))

        canonical_weights = 1.0 / (canonical_weights + 1e-6)
        canonical_volume = canonical_volume * canonical_weights
        canonical_sigma = canonical_sigma * canonical_weights

        # mask = canonical_weights > 0
        # weights = canonical_weights.clone()
        # weights[mask] = 1.0 / weights[mask]
        # canonical_volume = canonical_volume * weights

        self.density_volume = canonical_sigma
        self.volume = RefVolume(canonical_volume).to(device)

        del canonical_volume, canonical_weights
        torch.cuda.empty_cache()

    def update_density_volume(self):
        with torch.no_grad():
            print('update density')
            network_fn = self.render_kwargs_train['network_fn']
            network_query_fn = self.render_kwargs_train['network_query_fn']

            D,H,W = self.volume.feat_volume.shape[-3:]
            features = self.volume.feat_volume.permute(0,2,3,4,1).reshape(D*H,W,-1)
            self.density_volume = render_density(network_fn, self.vox_pts, features, network_query_fn).reshape(1,1,D,H,W)


    def decode_batch(self, batch):
        rays = batch['rays'].squeeze()  # (B, 8)
        rgbs = batch['rgbs'].squeeze()  # (B, 3)
        return rays, rgbs

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
        return (data - mean) / std

    def forward(self):
        return


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.grad_vars, lr=self.args.lrate, betas=(0.9, 0.999))
        scheduler = get_scheduler(self.args, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):

        rays, rgbs_target = self.decode_batch(batch)


        if args.N_importance and 0 == self.global_step%500:
            self.update_density_volume()

        xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays, N_samples=args.N_samples, N_importance=args.N_importance,
                                        lindisp=args.use_disp, perturb=args.perturb, density_volume=self.density_volume, bbox_3D=self.bbox_3d)

        # Converting world coordinate to ndc coordinate
        xyz_NDC = (xyz_coarse_sampled - self.bbox_3d[0].view(1,1,3))/(self.bbox_3d[1]-self.bbox_3d[0]).view(1,1,3)

        # rendering
        rgbs, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source_ref, xyz_coarse_sampled, xyz_NDC, z_vals, rays_o, rays_d,
                                                       self.volume, **self.render_kwargs_train)

        log, loss = {}, 0
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], rgbs_target)
            loss = loss + img_loss0
            psnr0 = mse2psnr2(img_loss0.item())
            self.log('train/PSNR0', psnr0.item(), prog_bar=True)


        ##################  rendering #####################
        if self.args.with_rgb_loss:
            img_loss = img2mse(rgbs, rgbs_target)
            loss += img_loss
            psnr = mse2psnr2(img_loss.item())

            with torch.no_grad():
                self.log('train/loss', loss, prog_bar=True)
                self.log('train/img_mse_loss', img_loss.item(), prog_bar=False)
                self.log('train/PSNR', psnr.item(), prog_bar=True)

        # if self.global_step == 3999 or self.global_step == 9999:
        #     self.save_ckpt(f'{self.global_step}')

        return  {'loss':loss}


    def validation_step(self, batch, batch_nb):

        self.MVSNet.train()
        rays, img = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        img = img.squeeze().cpu()  # (H, W, 3)


        N_rays_all = rays.shape[0]

        ##################  rendering #####################
        keys = ['val_psnr_all']
        log = init_log({}, keys)
        with torch.no_grad():

            rgbs, depth_preds = [],[]
            for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                    lindisp=args.use_disp, N_samples=args.N_samples, N_importance=args.N_importance, density_volume=self.density_volume, bbox_3D=self.bbox_3d)

                # Converting world coordinate to ndc coordinate
                xyz_NDC = (xyz_coarse_sampled - self.bbox_3d[0].view(1, 1, 3)) / (self.bbox_3d[1] - self.bbox_3d[0]).view(1, 1, 3)

                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source_ref, xyz_coarse_sampled,
                                                                       xyz_NDC, z_vals, rays_o, rays_d,
                                                                       self.volume,
                                                                       **self.render_kwargs_train)

                rgbs.append(rgb.cpu());depth_preds.append(depth_pred.cpu())

            H,W = img.shape[:2]
            rgbs, depth_r = torch.clamp(torch.cat(rgbs).reshape(H, W, 3),0,1), torch.cat(depth_preds).reshape(H, W)
            img_err_abs = (rgbs - img).abs()

            log['val_psnr_all'] = mse2psnr(torch.mean(img_err_abs ** 2))
            depth_r, _ = visualize_depth(depth_r, self.near_far_source)
            self.logger.experiment.add_images('val/depth_gt_pred', depth_r[None], self.global_step)

            img_vis = torch.stack((img, rgbs, img_err_abs.cpu()*5)).permute(0,3,1,2)
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step)
            os.makedirs(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/',exist_ok=True)

            img_vis = torch.cat((img,rgbs,img_err_abs*10,depth_r.permute(1,2,0)),dim=1).numpy()
            imageio.imwrite(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}'
                            f'/{self.args.expname}_{self.global_step:08d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))
            self.idx += 1

        return log

    def validation_epoch_end(self, outputs):

        if self.args.with_depth:
            mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
            mask_sum = torch.stack([x['mask_sum'] for x in outputs]).sum()
            # mean_d_loss_l = torch.stack([x['val_depth_loss_l'] for x in outputs]).mean()
            mean_d_loss_r = torch.stack([x['val_depth_loss_r'] for x in outputs]).mean()
            mean_abs_err = torch.stack([x['val_abs_err'] for x in outputs]).sum() / mask_sum
            mean_acc_1mm = torch.stack([x[f'val_acc_{self.eval_metric[0]}mm'] for x in outputs]).sum() / mask_sum
            mean_acc_2mm = torch.stack([x[f'val_acc_{self.eval_metric[1]}mm'] for x in outputs]).sum() / mask_sum
            mean_acc_4mm = torch.stack([x[f'val_acc_{self.eval_metric[2]}mm'] for x in outputs]).sum() / mask_sum

            self.log('val/d_loss_r', mean_d_loss_r, prog_bar=False)
            self.log('val/PSNR', mean_psnr, prog_bar=False)

            self.log('val/abs_err', mean_abs_err, prog_bar=False)
            self.log(f'val/acc_{self.eval_metric[0]}mm', mean_acc_1mm, prog_bar=False)
            self.log(f'val/acc_{self.eval_metric[1]}mm', mean_acc_2mm, prog_bar=False)
            self.log(f'val/acc_{self.eval_metric[2]}mm', mean_acc_4mm, prog_bar=False)

        mean_psnr_all = torch.stack([x['val_psnr_all'] for x in outputs]).mean()
        self.log('val/PSNR_all', mean_psnr_all, prog_bar=True)
        return


    def save_ckpt(self, name='latest'):
        save_dir = f'runs_fine_tuning/{self.args.expname}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            'global_step': self.global_step,
            'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            'volume': self.volume.state_dict(),
            'network_mvs_state_dict': self.MVSNet.state_dict()}
        if self.render_kwargs_train['network_fine'] is not None:
            ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = MVSSystem(args)
    checkpoint_callback = ModelCheckpoint(os.path.join(f'runs_fine_tuning/{args.expname}/ckpts/','{epoch:02d}'),
                                          monitor='val/PSNR',
                                          mode='max',
                                          save_top_k=0)

    logger = loggers.TestTubeLogger(
        save_dir="runs_fine_tuning",
        name=args.expname,
        debug=False,
        create_git_tag=False
    )

    args.num_gpus, args.use_amp = 1, False
    trainer = Trainer(max_epochs=args.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=args.num_gpus,
                      distributed_backend='ddp' if args.num_gpus > 1 else None,
                      num_sanity_val_steps=1, #if args.num_gpus > 1 else 5,
                      check_val_every_n_epoch = max(system.args.num_epochs//system.args.N_vis,1),
                      benchmark=True,
                      precision=16 if args.use_amp else 32,
                      amp_level='O1')

    trainer.fit(system)
    system.save_ckpt()
