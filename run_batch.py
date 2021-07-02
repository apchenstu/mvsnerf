import os

cuda = 2
N_samples = 128
N_importance = 0
batch_size = 1024

####################   train general model   #################################
cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train_mvs_nerf_pl.py --with_depth  --imgScale_test {1.0} ' \
      f'--expname mvs-nerf-net-raw --num_epochs {6} --batch_size {batch_size} --N_samples {N_samples} --use_viewdirs ' \
      f'--dataset_name dtu --datadir /mnt/data/new_disk/sungx/data/mvs_dataset/DTU/mvs_training/dtu --N_vis {6} --netwidth 256 --net_type v0 --pad 24'
print(cmd)
os.system(cmd)


##########################   nerf finetuning  ##################################
# for data_name in ['ship']:#'ship','mic','chair','lego','drums','ficus','materials','hotdog'
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train_mvs_nerf_finetuning_pl.py ' \
#           f'--dataset_name blender --datadir /mnt/new_disk_2/anpei/Dataset/nerf_synthetic/{data_name} '\
#           f'--expname {data_name}  --with_rgb_loss  --batch_size {batch_size} ' \
#           f'--num_epochs {1} --imgScale_test {1.0} --white_bkgd --N_samples {N_samples} --pad 0  ' \
#           f'--ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 '
#     print(cmd)
#     os.system(cmd)

##########################  LLFF finetuning  ##################################
# for data_name in ['horns']:#'fortress','leaves','trex''orchids','room','flower','horns','fern'
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train_mvs_nerf_finetuning_pl.py ' \
#           f'--dataset_name llff --datadir /mnt/new_disk_2/anpei/Dataset/MVSNeRF/nerf_llff_data/{data_name} '\
#           f'--expname {data_name}  --with_rgb_loss --batch_size {batch_size} --pad 24 ' \
#           f'--num_epochs {1}  --N_samples {N_samples}  ' \
#           f'--ckpt ./ckpts//mvsnerf-v0.tar --N_vis 1 '
#
#     print(cmd)
#     os.system(cmd)

# ##########################  DTU finetuning  ##################################
# for scene in [114]:#
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train_mvs_nerf_finetuning_pl.py ' \
#           f'--dataset_name dtu_ft --datadir /mnt/data/new_disk/sungx/data/mvs_dataset/DTU/mvs_training/dtu/scan{scene} '\
#           f'--expname dtu_scan{scene}  --with_rgb_loss  --batch_size {batch_size} ' \
#           f'--num_epochs {2} --N_vis {2} --imgScale_test {1.0}  --N_samples {N_samples} --pad 24 ' \
#           f'--ckpt ./ckpts//mvsnerf-v0.tar --N_vis 1'
#
#     print(cmd)
#     os.system(cmd)



##########################  LLFF finetuning old model  #############################
# for data_name in ['horns']:#'fortress','leaves','trex''orchids','room','flower','horns','fern'
#     cmd = f'CUDA_VISIBLE_DEVICES={cuda}  python train_mvs_nerf_finetuning_pl.py ' \
#           f'--dataset_name llff --datadir /mnt/new_disk_2/anpei/Dataset/MVSNeRF/nerf_llff_data/{data_name} '\
#           f'--expname {data_name}  --with_rgb_loss --batch_size {batch_size} --pad 24 --netwidth 128 --net_type v0 ' \
#           f'--num_epochs {1}  --N_samples {N_samples}  ' \
#           f'--ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 '
#
#     print(cmd)
#     os.system(cmd)
