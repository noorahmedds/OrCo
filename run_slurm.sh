#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH --gres gpu:1
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/slurm-%A_%a.out

cmd="python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 1 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_simclr01_convexcomb_pert_offset -perturb_offset 0.5"
cmd="python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -save_path_prefix autoaugment -init_sess_w_base_proj -perturb_mode inc+curr-base -rand_aug_sup_con"
cmd="python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix best_ver2_projtype+jointxent -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -decay 5e-4 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 0.025 -lr_base_encoder 0.025 -base_schedule Cosine -proj_type proj -init_sess_w_base_proj -perturb_mode inc+curr-base"


# print start time and command to log
# Use tail -1 to read the logs
echo $(date)
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo $cmd


# start command
$cmd