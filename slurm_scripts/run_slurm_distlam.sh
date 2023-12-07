#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 15:00:00
#SBATCH --gres gpu:1
#SBATCH -a 0-9
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/slurm-%A_%a.out
 

PARRAY=(1 2 4 6 8 10 15 20 30 50)

#p1 is the element of the array found with ARRAY_ID mod P_ARRAY_LENGTH
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

cmd="python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -save_path_prefix proj_ema_exp -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir  /BS/fscil/work/code/solo-learn/trained_models/supcon/s7rclcq9/pretrain_cub_aa_epoch480_81_183.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -novel_bias -lr_base 0.25 -proj_ema_update -proj_ema_update_every $p1 -proj_ema_mode all -resume -dist_lam 0.6"

echo $(date)
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo $cmd

# start command
$cmd