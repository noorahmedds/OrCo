model_dir="./params/OrCo/cifar100/cifar_e280_01simclr_acc80_22.ckpt"

python "train.py" \
    -project orco \
    -dataset cifar100 \
    -save_path_prefix best \
    -gpu 0 \
    -model_dir ${model_dir} \
    -epochs_base 10 \
    -epochs_joint 100 \
    -batch_size_joint 64 \
    -joint_schedule Cosine \
    -decay 5e-4 \
    -decay_new 1e-4 \
    -cos_b_lam 0 \
    -lr_base 0.25 \
    -perturb_epsilon_base 9e-05 \
    -perturb_epsilon_inc 9e-05 \
    -rand_aug_sup_con