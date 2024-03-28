python train.py \
    -project orco \
    -dataset cub200 \
    -save_path_prefix best \
    -gpu 0 \
    -epochs_base 100 \
    -epochs_joint 100 \
    -epochs_target_gen 5000 \
    -batch_size_joint 32 \
    -base_schedule Cosine \
    -joint_schedule Cosine \
    -reserve_mode full \
    -decay 5e-4 \
    -decay_new 1e-5 \
    -cos_b_lam 0.6 \
    -lr_base 0.025 \
    -lr_base_encoder 0.025 \
    -lr_new 0.05 \
    -fine_tune_backbone_base \