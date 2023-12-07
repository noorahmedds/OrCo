# # mini_imagenet for confusion matrix
# python train.py -project base -dataset mini_imagenet -model_dir ./checkpoint/base_training_ckpt/mini_imagenet/base/ft_dot/session0_70_500.pth -start_session 1 -gpu 2 -new_mode ft_dot -epochs_new 150 -base_mode ft_dot -save_path_prefix "exp_mini_imagenet_conf_mat"

# # Cub200
# python train.py -project base -dataset cub200 -model_dir checkpoint/base_training_ckpt/cub200/base/ft_dot/session0_70_976.pth -start_session 1 -gpu 2 -new_mode ft_dot -epochs_new 150 -base_mode ft_dot -save_path_prefix "exp_cub200_conf_mat"

# python train.py -project base_infonce -dataset mini_imagenet -model_dir ./checkpoint/base_training_ckpt/mini_imagenet/base/ft_dot/session0_70_500.pth -gpu 1 -start_session 1 -save_path_prefix "exp_miniimagenet_infonce_log" -epochs_new 150


# # CUB 200 label smoothing
# python train.py -project base -dataset cub200 -gpu 0 -base_mode ft_dot -new_mode ft_dot -epochs_base 200 -epochs_new 150 -save_path_prefix "exp_cub200_labelsmoothing_0.5" -label_smoothing 0.5

# python train.py -project base -dataset cub200 -start_session 1 -gpu 2 -new_mode ft_dot -epochs_new 150 -save_path_prefix "exp_corrected_joint_session" -model_dir "./checkpoint/base_training_ckpt/cub200/base/ft_dot/session0_70_976.pth" -label_smoothing 0.5

python train.py -project base -dataset cifar100 -start_session 0 -gpu 2 -new_mode ft_dot -epochs_new 150 -save_path_prefix "exp_corrected_joint_session"