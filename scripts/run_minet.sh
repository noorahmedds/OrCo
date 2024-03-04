python "train.py" \
    -project orco \
    -dataset mini_imagenet \
    -start_session 1 \
    -gpu 0 \
    -batch_size_joint 64 \
    -train_inter \
    -epochs_joint 100 \
    -model_dir /BS/fscil/work/code/solo-learn/trained_models/supcon/muqbelec/minet_e380_01simclr_acc85_95.ckpt \
    -reserve_mode all \
    -epochs_base 10 \
    -pull_criterion_novel xent \
    -pull_criterion_base xent \
    -joint_supcon \
    -validation_metric hm \
    -joint_schedule Cosine \
    -decay_new 1e-4 \
    -decay 1e-4 \
    -cos_b_lam 0 \
    -init_sess_w_base_proj \
    -perturb_mode inc+curr-base \
    -sup_lam 1 \
    -cos_lam 1 \
    -simplex_lam 1 \
    -save_path_prefix minet \
    -perturb_offset 0.5
    # -sup_con_pretrain \

# Args to remove: start_session (assign default value), supcon pretrain is always on, reserve mode default, joint supcon can be removed
# pull criterion can be removed
# perturb_mode default 
# Use default weights for sup, cos, simplex
# perturb offset is default
