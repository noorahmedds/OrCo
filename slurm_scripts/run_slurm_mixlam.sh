#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 15:00:00
#SBATCH --gres gpu:1
#SBATCH -a 0-7
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/slurm_mix_lam-%A_%a.out
 

PARRAY=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

#p1 is the element of the array found with ARRAY_ID mod P_ARRAY_LENGTH
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

cmd="python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix no_pretrain_mixup_lam_$p1 -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -novel_bias -proj_type proj -lr_base 0.025 -decay 5e-4 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -batch_size_base 256 -mix_lam $p1"

echo $(date)
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo $cmd

# start command
$cmd