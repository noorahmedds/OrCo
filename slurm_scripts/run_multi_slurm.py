import tqdm
from sklearn.model_selection import ParameterGrid
import subprocess
import os

partition='gpu20'
days = '0'
hours = 6
gpu = 0

# command_group = "reruns_correct_pert_cub"
# run_commands = [
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -simplex_lam 1 -novel_bias -perturb_offset 0.5", 
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -skip_orth",
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -cos_lam 0 -simplex_lam 0 -sup_lam 1",   
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 0",
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 0 -simplex_lam 1",
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -simplex_lam 1 -novel_bias -exemplars_count 1",
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -simplex_lam 1 -novel_bias -exemplars_count 5",
#     # "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 5 -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -simplex_lam 1 -novel_bias -exemplars_count 0 -perturb_mode all"
#     # "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -novel_bias -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05",
#     # "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -cos_b_lam 0 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode inc+curr-base "
#     "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -novel_bias -decay 5e-4 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 0.025 -lr_base_encoder 0.025 -base_schedule Cosine -proj_type proj",
#     "python train.py -project base_supcon_srv3 -dataset cub200 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 32 -train_inter -epochs_joint 100 -reserve_mode full -epochs_base 100 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-5 -cos_b_lam 0.6 -decay 5e-4 -epochs_simplex 5000 -lr_new 0.05 -simplex_lam 1 -fine_tune_backbone_base -lr_base 0.025 -lr_base_encoder 0.025 -base_schedule Cosine -proj_type proj -init_sess_w_base_proj -perturb_mode inc+curr-base"
# ]

# command_group = "pert_pull_search"
# run_commands = [
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_all+ua_pull_inc -perturb_offset 0.5 -perturb_mode all -cos_b_lam 0",
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_all-ua_pull_inc -perturb_offset 0.5 -perturb_mode all-ua -cos_b_lam 0",
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_inc-ua_pull_inc -perturb_offset 0.5 -perturb_mode inc+curr-base-ua -cos_b_lam 0",
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_inc+ua_pull_all -perturb_offset 0.5 -perturb_mode inc+curr-base -cos_b_lam 0.5",
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_all+ua_pull_all -perturb_offset 0.5 -perturb_mode all -cos_b_lam 0.5",
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_all-ua_pull_all -perturb_offset 0.5 -perturb_mode all-ua -cos_b_lam 0.5",
#     "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -init_sess_w_base_proj -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_pert_inc-ua_pull_all -perturb_offset 0.5 -perturb_mode inc+curr-base-ua -cos_b_lam 0.5"
# ]

# command_group = "pert_pull_search_CIFAR"
# run_commands = [
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode all -cos_b_lam 0",
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode all-ua -cos_b_lam 0",
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode inc+curr-base-ua -cos_b_lam 0",
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode inc+curr-base -cos_b_lam 0.5",
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode all -cos_b_lam 0.5",
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode all-ua -cos_b_lam 0.5",
#     "python train.py -project base_supcon_srv3 -dataset cifar100 -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/15dtcwvu/supcon+simclr+280+80_22.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 5e-4 -lr_base 0.25 -simplex_lam 1 -perturb_epsilon_base 9e-05 -perturb_epsilon_inc 9e-05 -init_sess_w_base_proj -perturb_mode inc+curr-base-ua -cos_b_lam 0.5"
# ]

command_group = "simclr_wight"
run_commands = [
    "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/kvsqbgmy/minet_epochs380_0_2simclr_acc_85_667.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_simclr02_convexcomb",
    "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/sr2f28p8/minet_simclr0_3_acc_85_35.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_simclr03_convexcomb",
    "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/oby37s09/minet_simclr0_4_ep380_acc_84_667.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_simclr04_convexcomb",
    "python train.py -project base_supcon_srv3 -dataset mini_imagenet -start_session 1 -gpu 0 -sup_con_pretrain -batch_size_joint 64 -train_inter -epochs_joint 100 -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/f8a5f6ko/minet_simclr0_5_ep380_acc_84_25.ckpt -reserve_mode all -epochs_base 10 -pull_criterion_novel xent -pull_criterion_base xent -joint_supcon -validation_metric hm -joint_schedule Cosine -decay_new 1e-4 -decay 1e-4 -cos_b_lam 0 -init_sess_w_base_proj -perturb_mode inc+curr-base -sup_lam 1 -cos_lam 1 -simplex_lam 1 -save_path_prefix best_minet_simclr05_convexcomb"
]


command_group_logs_dir = f"/BS/fscil/work/code/CEC-CVPR2021/logs/{command_group}"
if not os.path.exists(command_group_logs_dir):
    os.mkdir(command_group_logs_dir)

# Run for each element in the parameter grid
for ix, exp in tqdm.tqdm(enumerate(run_commands)):
    # # Make save path prefix for this run
    run_save_path_prefix = f"-save_path_prefix {command_group}_{ix}"

    command = f"{exp} {run_save_path_prefix}"
    
    # print(command)
    
    script_content = f"""#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 6:00:00
#SBATCH --gres gpu:1
#SBATCH -o /BS/fscil/work/code/CEC-CVPR2021/logs/{command_group}/slurm-%A_{command_group}_{ix}.out

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

