from opt import config_parser
from torch.utils.data import DataLoader

from data import dataset_dict

# models
from models import *
from renderer import *
from utils import *
from data.ray_utils import ray_marcher,ray_marcher_fine

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

class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
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
        self.init_volume()
        self.grad_vars += list(self.volume.parameters())


    def init_volume(self):

        self.imgs, self.proj_mats, self.near_far_source, self.pose_source = self.train_dataset.read_source_views(device=device)
        ckpts = torch.load(args.ckpt)
        if 'volume' not in ckpts.keys():
            self.MVSNet.train()
            with torch.no_grad():
                volume_feature, _, _ = self.MVSNet(self.imgs, self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
        else:
            volume_feature = ckpts['volume']['feat_volume']
            print('load ckpt volume.')
        self.imgs = self.unpreprocess(self.imgs)

        # project colors to a volume
        self.density_volume = None
        if args.use_color_volume or args.use_density_volume:
            D,H,W = volume_feature.shape[-3:]
            intrinsic, c2w = self.pose_source['intrinsics'][0].clone(), self.pose_source['c2ws'][0]
            intrinsic[:2] /= 4
            vox_pts = get_ptsvolume(H-2*args.pad,W-2*args.pad,D, args.pad,  self.near_far_source, intrinsic, c2w)

            self.color_feature = build_color_volume(vox_pts, self.pose_source, self.imgs, with_mask=True).view(D,H,W,-1).unsqueeze(0).permute(0, 4, 1, 2, 3)  # [N,D,H,W,C]
            if args.use_color_volume:
                volume_feature = torch.cat((volume_feature, self.color_feature),dim=1) # [N,C,D,H,W]

            if args.use_density_volume:
                self.vox_pts = vox_pts

            else:
                del vox_pts

        self.volume = RefVolume(volume_feature.detach()).to(device)
        del volume_feature

    def update_density_volume(self):
        with torch.no_grad():
            network_fn = self.render_kwargs_train['network_fn']
            network_query_fn = self.render_kwargs_train['network_query_fn']

            D,H,W = self.volume.feat_volume.shape[-3:]
            features = torch.cat((self.volume.feat_volume, self.color_feature), dim=1).permute(0,2,3,4,1).reshape(D*H,W,-1)
            self.density_volume = render_density(network_fn, self.vox_pts, features, network_query_fn).reshape(D,H,W)
        del features

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

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=args.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):

        rays, rgbs_target = self.decode_batch(batch)

        if args.use_density_volume and 0 == self.global_step%200:
            self.update_density_volume()

        xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays, N_samples=args.N_samples,
                        lindisp=args.use_disp, perturb=args.perturb)

        # Converting world coordinate to ndc coordinate
        H,W = self.imgs.shape[-2:]
        inv_scale = torch.tensor([W - 1, H - 1]).to(device)
        w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0]
        xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale, near=self.near_far_source[0],far=self.near_far_source[1], pad=args.pad, lindisp=args.use_disp)

        # important sampleing
        if self.density_volume is not None and args.N_importance > 0:
            xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(rays, self.density_volume, z_vals, xyz_NDC,
                                                                          N_importance=args.N_importance)
            xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                         near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad, lindisp=args.use_disp)

        # rendering
        rgbs, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled, xyz_NDC, z_vals, rays_o, rays_d,
                                                       self.volume, self.imgs,  **self.render_kwargs_train)

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
        img = img.cpu()  # (H, W, 3)
        # mask = batch['mask'][0]

        N_rays_all = rays.shape[0]

        ##################  rendering #####################
        keys = ['val_psnr_all']
        log = init_log({}, keys)
        with torch.no_grad():

            rgbs, depth_preds = [],[]
            for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                    N_samples=args.N_samples, lindisp=args.use_disp)

                # Converting world coordinate to ndc coordinate
                H, W = img.shape[:2]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0].clone()
                intrinsic_ref[:2] *= args.imgScale_test/args.imgScale_train
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad*args.imgScale_test, lindisp=args.use_disp)

                # important sampleing
                if self.density_volume is not None and args.N_importance > 0:
                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                    self.density_volume, z_vals,xyz_NDC,N_importance=args.N_importance)
                    xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                    near=self.near_far_source[0], far=self.near_far_source[1],pad=args.pad, lindisp=args.use_disp)


                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled,
                                                                       xyz_NDC, z_vals, rays_o, rays_d,
                                                                       self.volume, self.imgs,
                                                                       **self.render_kwargs_train)

                rgbs.append(rgb.cpu());depth_preds.append(depth_pred.cpu())


            rgbs, depth_r = torch.clamp(torch.cat(rgbs).reshape(H, W, 3),0,1), torch.cat(depth_preds).reshape(H, W)
            img_err_abs = (rgbs - img).abs()

            log['val_psnr_all'] = mse2psnr(torch.mean(img_err_abs ** 2))
            depth_r, _ = visualize_depth(depth_r, self.near_far_source)
            self.logger.experiment.add_images('val/depth_gt_pred', depth_r[None], self.global_step)

            img_vis = torch.stack((img, rgbs, img_err_abs.cpu()*5)).permute(0,3,1,2)
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step)
            os.makedirs(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/',exist_ok=True)

            img_vis = torch.cat((img,rgbs,img_err_abs*10,depth_r.permute(1,2,0)),dim=1).numpy()
            imageio.imwrite(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/{self.global_step:08d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))
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
                      # check_val_every_n_epoch = max(system.args.num_epochs//system.args.N_vis,1),
                      val_check_interval=500,
                      benchmark=True,
                      precision=16 if args.use_amp else 32,
                      amp_level='O1')

    trainer.fit(system)
    system.save_ckpt()
