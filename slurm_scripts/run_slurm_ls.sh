#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 15:00:00
#SBATCH --gres gpu:1
#SBATCH -a 0-2
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/slurm-%A_%a.out
 

PARRAY=(0.90 0.95 0.99)

#p1 is the element of the array found with ARRAY_ID mod P_ARRAY_LENGTH
p1=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}

cmd="python train.py -project base -dataset cub200 -start_session 0 -epochs_base 200 -gpu 0 -new_mode ft_dot -epochs_new 150 -save_path_prefix exp_ls_search -label_smoothing $p1"

echo $(date)
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
echo $cmd

# start command
$cmd