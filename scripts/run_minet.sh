python "train.py" \
    -project orco \
    -dataset mini_imagenet \
    -save_path_prefix best \
    -gpu 0 \
    -model_dir ./params/OrCo/minet/minet_e380_01simclr_acc85_95.ckpt \
    -epochs_base 10 \
    -epochs_joint 100 \
    -batch_size_joint 64 \
    -joint_schedule Cosine \
    -decay_new 1e-4 \
    -decay 1e-4 \
    -cos_b_lam 0 \