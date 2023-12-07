import tqdm
from sklearn.model_selection import ParameterGrid
import subprocess
import os

partition='gpu20'
days = '0'
hours = 2
gpu = 0

# ----------- lr, wd  hyperparam search -----------
# adam
# lrs = (0.5e-5, 1e-5, 2e-5)
# wds = (0.1e-6, 0.5e-6, 1e-6, 0.1e-5, 0.1e-4)

#adamw
# epochs = (60,) #60, 80, 100)
# lrs = (5e-4, 2e-4, 1e-4)
# wds = (0.1, 0.05, 0.02)

# ===== Adam For cub without pretraining (running only the preparatory step) 
# param_grid = [{
#     'lr_base': [0.5e-5, 1e-5, 1e-4, 1e-3], 
#     'optimizer': ["adam"],
#     'decay': [0.1e-6, 0.5e-6, 1e-6]
#     }]

# ===== AdamW For cub without pretraining (running only the preparatory step) 
# param_grid = [{
#     'lr_base': [5e-4, 2e-4, 1e-4], 
#     'optimizer': ["adamw"],
#     'decay': [0.1, 0.05, 0.02]
#     }]
# inheriting_params = {"lr_base":"lr_base_encoder"}
# base_command = "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -novel_bias -proj_type proj -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -base_schedule Cosine -end_session 1" # -optimizer sgd -optimizer_joint adam -lr_new 3e-4 -lr_base 0.025 -lr_base_encoder 0.025"
# save_path_prefix = "cub_optimizer_ablation_only_base"

# ===== CIFAR100 Hp search
# param_grid = [{
#     'batch_size_joint': [16, 32, 64, 128],
#     'simplex_lam': [0.1, 0.5, 1],
#     "simplex_lam_inc": [0, 0.1, 0.5, 1],
#     'cos_lam': [0.1, 0.5, 1],
#     "sup_lam": [0.5, 0.7, 1.0],
#     "cos_b_lam": [0.01, 0.1, 0.2],
#     "cos_n_lam": [0.5, 0.7, 1.0],
#     "sup_lam": [0.5, 0.7, 1.0],
#     }]
# inheriting_params = {}
# base_command = "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -train_inter -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/s7rclcq9/pretrain_cub_aa_epoch480_81_183.ckpt -reserve_mode all -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -novel_bias -epochs_base 10 -lr_base 0.25 -lr_new 0.1 -cos_b_lam 0 -epochs_joint 100 -simplex_lam 1"
# save_path_prefix = "cifar_lam_search"

# ===== CIFAR100 different pretrain
# param_grid = [{
#     'lr_base': [0.2, 0.1, 0.05],
#     'lr_new': [0.2, 0.1, 0.05], 
#     'decay': [5e-4, 1e-4, 5e-3],
#     'decay_new': [5e-4, 1e-4, 5e-3],
#     # 'batch_size_joint': [32, 64, 128],
#     'epochs_base': [10, 20],
#     }]
# base_command = "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -save_path_prefix different_pretrain -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir  /BS/fscil/work/code/SupCon-Framework/weights/supcon_first_stage_cifar100resnet12/epoch96 -reserve_mode all -epochs_base 10 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -cos_b_lam 0 -novel_bias -simplex_lam 1" #-decay_new 1e-4 -decay 5e-4

# # ==== CIFAR100 decay search
# param_grid = [{
#     'batch_size_joint': [16, 32, 64, 128],
#     'simplex_lam': [0.1, 0.5, 1],
#     "simplex_lam_inc": [0, 0.1, 0.5, 1],
#     'cos_lam': [0.1, 0.5, 1],
#     "sup_lam": [0.5, 0.7, 1.0],
#     "cos_b_lam": [0.01, 0.1, 0.2],
#     "cos_n_lam": [0.5, 0.7, 1.0],
#     "sup_lam": [0.5, 0.7, 1.0],
#     }]
# inheriting_params = {}
# base_command = "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -train_inter -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/s7rclcq9/pretrain_cub_aa_epoch480_81_183.ckpt -reserve_mode all -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -novel_bias -epochs_base 10 -lr_base 0.25 -lr_new 0.1 -cos_b_lam 0 -epochs_joint 100 -simplex_lam 1"
# save_path_prefix = "cifar_lam_search"

# ==== CIFAR100 epsilon search
# param_grid = [{
#     "perturb_epsilon_base": [0.1, 5e-2, 1e-2, 1e-3],
#     "perturb_epsilon_inc": [0.1, 5e-2, 1e-2, 1e-3],
#     }]
# inheriting_params = {}
# base_command = "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir  /BS/fscil/work/code/solo-learn/trained_models/supcon/s7rclcq9/pretrain_cub_aa_epoch480_81_183.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -novel_bias -lr_base 0.25 -simplex_lam 1"
# save_path_prefix = "cifar_lam_search"

# ==== CIFAR100 perturbation search
# param_grid = [{
#     "perturb_epsilon_base": [1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0],
#     # "perturb_epsilon_base": [9e-5, 7e-5, 5e-5, 1e-5, 1e-6]
# }]
# inheriting_params = {"perturb_epsilon_base":"perturb_epsilon_inc"}
# # base_command = "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -lr_base 0.25 -simplex_lam 1 -init_sess_w_base_proj -perturb_mode inc+curr-base -rand_aug_sup_con -batch_size_perturb 0"
# # save_path_prefix = "cifar_pert_search_rerun_nobs"
# base_command = "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel cosine -pull_criterion_base cosine -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -lr_base 0.25 -simplex_lam 1 -init_sess_w_base_proj -perturb_mode inc+curr-base -rand_aug_sup_con -batch_size_perturb 0"
# save_path_prefix = "cifar_pert_search_rerun_nobs_cosine"


# ===== Reserve vector count search
param_grid = [{
    "reserve_vector_count": [200, 300, 400, 500, 600, 700, 800, 900],
}]
inheriting_params = {}
base_command="python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -perturb_offset 0.5"
save_path_prefix = "minet_reserve_vector_count_abl"

# ==== Minet perturbation search
# param_grid = [{
#     # "perturb_epsilon_base": [0.5, 0.1, 5e-2, 1e-3, 5e-3, 1e-4, 1e-5, 1e-6]
#     "perturb_epsilon_base": [0.0, 1.0]
#     # "perturb_epsilon_base": [5e-2, 1e-3]
# }]
# inheriting_params = {"perturb_epsilon_base":"perturb_epsilon_inc"}
# base_command = "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1"
# save_path_prefix = "minet_pert_search_rerun"

# -----

# mkdir
logs_dir = f"/BS/fscil/work/code/CEC-CVPR2021/logs/{save_path_prefix}"
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)

# Generate the exhaustive grid search
grid = ParameterGrid(param_grid)
# Run for each element in the parameter grid
for exp in tqdm.tqdm(grid):
    # Create command extension
    command_extension = ""
    for k,v in exp.items():
        command_extension += f"-{k} {v} "

    # Add to command any values which inherit from other values in the param_grid
    for k,v in inheriting_params.items():
        command_extension += f"-{v} {exp[k]}"
    
    # Make save path prefix for this run
    run_save_path_prefix = f"-save_path_prefix {save_path_prefix}+{command_extension.replace('-', '').replace(' ', '-')}"
    
    # Complete the final command
    command = f"{base_command} {command_extension} {run_save_path_prefix}"
    
    # print(command)
    
    script_content = f"""#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 6:00:00
#SBATCH --gres gpu:1
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/{save_path_prefix}/slurm-%A_{command_extension.replace('-', '').replace(' ', '-')}.out

cmd="{command}"

echo $(date)
echo $cmd

$cmd"""

    # Create new bash file
    with open('run_temp.sh', 'w') as fout:
        fout.write(script_content)

    # Run the command
    bashCommand = f"sbatch ./run_temp.sh"
    process = subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
    output, error = process.communicate()
    print(output)
    print(error)

