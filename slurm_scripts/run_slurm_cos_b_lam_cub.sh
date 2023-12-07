#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 15:00:00
#SBATCH --gres gpu:1
#SBATCH -a 0-9
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/slurm-%A_%a.out
 

PARRAY=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

#p1 is the element of the array found with ARRAY_ID mod P_ARRAY_LENGTH
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

cmd="python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -save_path_prefix test_joint_best+full -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir  /BS/fscil/work/code/solo-learn/trained_models/supcon/ep0hcst3/pretrain_cub_augmix_78_81.ckpt -reserve_mode full -epochs_base 30 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -cos_b_lam $p1 -novel_bias -proj_type proj_ncfscil -lr_base 0.025 -decay 5e-4 -epochs_simplex 5000 -resume"

echo $(date)
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo $cmd

# start command
$cmd