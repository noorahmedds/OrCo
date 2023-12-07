#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 6:00:00
#SBATCH --gres gpu:1
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/minet_reserve_vector_count_abl/slurm-%A_reserve_vector_count-900-.out

cmd="python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -perturb_offset 0.5 -reserve_vector_count 900  -save_path_prefix minet_reserve_vector_count_abl+reserve_vector_count-900-"

echo $(date)
echo $cmd

$cmd