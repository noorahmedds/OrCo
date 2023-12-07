#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 15:00:00
#SBATCH --gres gpu:1
#SBATCH -a 0-3
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/slurm-%A_%a.out
 

# PARRAY=(0 1 2 3 4 5)

jobArray=(
    "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain+lower_lr+cosesched+proj+inc_adam -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -cos_b_lam 0.6 -novel_bias -proj_type proj -decay 5e-4 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 0.025 -lr_base_encoder 0.025 -base_schedule Cosine -optimizer sgd -optimizer_joint adam -lr_new 3e-4 -decay_new 1e-6"
    "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain+lower_lr+cosesched+proj+inc_adamw -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -cos_b_lam 0.6 -novel_bias -proj_type proj -decay 5e-4 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 0.025 -lr_base_encoder 0.025 -base_schedule Cosine -optimizer sgd -optimizer_joint adamw -lr_new 3e-4 -decay_new 1e-1"
    # "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain+lower_lr+cosesched+proj+base_adam -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -novel_bias -proj_type proj -decay 1e-6 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 3e-4 -lr_base_encoder 3e-4 -base_schedule Cosine -optimizer adam -optimizer_joint sgd"
    # "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain+lower_lr+cosesched+proj+base_adamw -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -novel_bias -proj_type proj -decay 1e-1 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 3e-4 -lr_base_encoder 3e-4 -base_schedule Cosine -optimizer adamw -optimizer_joint sgd"
    # "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain+lower_lr+cosesched+proj+base_adam+inc_adam -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -cos_b_lam 0.6 -novel_bias -proj_type proj -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 3e-4 -lr_base_encoder 3e-4 -lr_new 3e-4 -base_schedule Cosine -optimizer adam -optimizer_joint adam -decay 1e-6 -decay_new 1e-6"
    # "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain+lower_lr+cosesched+proj+base_adamw+inc_adamw -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -cos_b_lam 0.6 -novel_bias -proj_type proj -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 3e-4 -lr_base_encoder 3e-4 -lr_new 3e-4 -base_schedule Cosine -optimizer adamw -optimizer_joint adamw -decay 1e-1 -decay_new 1e-1"
)

# Running all the cmds in parallel
cmd=${jobArray[`expr $SLURM_ARRAY_TASK_ID % ${#jobArray[@]}`]}

echo $(date)
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo $cmd

# start command
$cmd
