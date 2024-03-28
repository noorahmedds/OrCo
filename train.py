import argparse
import importlib
from utils import *

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-project', type=str, default="orco")
    parser.add_argument('-dataset', type=str, default='mini_imagenet',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default="../datasets/")
    parser.add_argument('-save_path_prefix', "-prefix", type=str, default="")

    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)

    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, help='loading model parameter from a specific dir')

    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_joint', type=int, default=100)
    parser.add_argument('-epochs_target_gen', type=int, default=1000)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_base_encoder', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1) 
    parser.add_argument('-optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', "mtadam"])
    parser.add_argument('-optimizer_joint', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', "mtadam"])    
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])

    parser.add_argument('-reserve_mode', type=str, default='all', 
                        choices=["all", "full"]) 

    parser.add_argument('-joint_schedule', type=str, default='Milestone',
                        choices=['Milestone', 'Cosine'])
    parser.add_argument('-base_schedule', type=str, default='Milestone',
                        choices=['Milestone', 'Cosine'])

    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-decay_new', type=float, default=0) 

    parser.add_argument('-cos_n_lam', type=float, default=0.5) 
    parser.add_argument('-cos_b_lam', type=float, default=0.5) 

    parser.add_argument('-sup_lam', type=float, default=1)
    parser.add_argument('-cos_lam', type=float, default=1)
    parser.add_argument('-simplex_lam', type=float, default=1)

    parser.add_argument('-perturb_epsilon_base', type=float, default=1e-2)
    parser.add_argument('-perturb_epsilon_inc', type=float, default=1e-2)
    parser.add_argument('-perturb_offset', type=float, default=0.5)

    parser.add_argument('-base_mode', type=str, default='ft_dot',
                        choices=['ft_dot', 'ft_cos', 'ft_l2', "ft_dot_freeze"]) 
    parser.add_argument('-fine_tune_backbone_base', action='store_true', help='')
    parser.add_argument("-proj_type", type=str, default="proj",
                        choices=["proj", "proj_ncfscil"])


    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_sup_con', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-batch_size_joint', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-batch_size_perturb', type=int, default=-1)
    parser.add_argument('-test_batch_size', type=int, default=100) 
    parser.add_argument('-drop_last_batch', action="store_true", help="Drops the last batch if not equal to the assigned batch size")
    parser.add_argument('-exemplars_count', type=int, default=-1)


    parser.add_argument('-rand_aug_sup_con', action='store_true', help='')
    parser.add_argument('-prob_color_jitter', type=float, default=0.8)
    parser.add_argument('-min_crop_scale', type=float, default=0.2)

    parser.add_argument('-warmup_epochs_base', type=int, default=3)
    parser.add_argument('-warmup_epochs_inc', type=int, default=10)

    # --- Other Params ---

    # parser.add_argument('-test', action="store_true", help="Rerun the testing for the experiments")
    # parser.add_argument('-experiment_path', type=str, default="")
    # parser.add_argument('-max_inference', action="store_true", help="Run the new inference scheme and compute the accuracies")
    # parser.add_argument('-t_scaling', action="store_true", help="run temperature scaling after the base session")
    # parser.add_argument('-base_only', action="store_true", help="cross entropy with only base classes in the denominator")
    # parser.add_argument('-balanced_testing', action="store_true", help="Do Balanced testing")

    # # dino
    # parser.add_argument('-all_crops', action="store_true", help="")
    # parser.add_argument('-dino_outdim', type=int, default=60000)
    # parser.add_argument('-dino_loss_weight', type=float, default=0.1)
    # parser.add_argument('-dino_last_layer', type=str, default='nll',
    #                     choices=['nll', 'unit', 'etf'])
    # parser.add_argument('-use_multi_opt', action="store_true", help="")
    # parser.add_argument('-dino_novel', action="store_true", help="")
    # parser.add_argument('-dino_joint', action="store_true", help="")
    # parser.add_argument('-class_level', action="store_true", help="Class level input for DINO vs instance level")
    # parser.add_argument('-two_step', action="store_true", help="EASY, two step forward backward pass at each step")
    # parser.add_argument('-cl2n', action="store_true", help="Feature transformation, centering l2 norm")
    # parser.add_argument('-dino_over_fc', action="store_true", help="Feature transformation, centering l2 norm")
    # parser.add_argument('-replace_mode', type=str, default='avg_cos', # CEC -> ft_cos
    #                     choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_l2', 'dot']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    
    # parser.add_argument('-mixup_start_epoch', type=int, default=100)
    # parser.add_argument('-s2m2', action="store_true", help="applys2m2 training schedule for the base session")
    # parser.add_argument('-classifier_last_layer', type=str, default='projection',
    #                     choices=['linear', 'projection'])
    # parser.add_argument('-classifier_hidden_dim', type=int, default=2048)
    # parser.add_argument('-vis_encoder', action="store_true", help="Visualise encoder ")

    # # Gaussian generator
    # parser.add_argument('-use_vector_variance', action="store_true", help="Use vector variance for gaussian generation")
    # parser.add_argument('-use_gaussian', action="store_true", help="Use the gaussian generated data for the joint session")
    # parser.add_argument('-views_gaussian', type=int, default = 1)
    # parser.add_argument('-aug_gaussian', action="store_true", help="augment the gaussian generated features by mixing them with other features from the same distribution")

    # # ==== Scheduling parameters
    # # teacher student temps
    # parser.add_argument('-student_temp', type=float, default=0.1)
    # parser.add_argument('-teacher_temp', type=float, default=0.07)

    # # teacher temperature scheduling
    # parser.add_argument('-teacher_scheduling', action="store_true", help="Scheudling the teacher temperature")
    # parser.add_argument('-warmup_teacher_temp', type=float, default=0.04)
    # parser.add_argument('-warmup_teacher_temp_epochs', type=int, default=25)
    # # lr scheduling
    # parser.add_argument('-schedule_lr', action="store_true", help="Cosine Scedhuling the lr ")
    # parser.add_argument('-min_lr', type=float, default=0.0048)
    # parser.add_argument('-warmup_epochs', type=int, default=10)
    # # wd scheduling
    # parser.add_argument('-schedule_wd', action="store_true", help="Cosine Scedhuling the wd")
    # parser.add_argument('-weight_decay_end', type=float, default=0.0005)
    # # teacher momentum scheduling
    # parser.add_argument("-momentum_teacher", type=float, default=0.996)
    
    # # contrast
    # parser.add_argument('-ce_pretrain', action="store_true", help="Pretrain on supcon")      
    # # parser.add_argument('-sup_con_pretrain', action="store_true", help="Pretrain on supcon")                                                    # True
    # parser.add_argument('-sup_con_base', action="store_true", help="Apply supcon in a multi task fashion during the base session")              # False
    # parser.add_argument('-use_sup_con_head', action="store_true", help="Have a projection mlp before the sup con loss for encoding projection") # True
    # parser.add_argument('-sup_con_feat_dim', type=int, default=128)                                                                             # 128
    # parser.add_argument('-sup_con_freeze_backbone', action="store_true", help="freeze Backbone after pretraining on sup con")                                                                        # True
    # parser.add_argument('-sup_con_temp', type=float, default=0.07)
    # parser.add_argument('-epochs_sup_con', type=int, default=50)                                                                             # 128
    # parser.add_argument('-lr_sup_con', type=float, default=0.05)
    # parser.add_argument('-momentum_sup_con', type=float, default=0.9)
    # parser.add_argument('-decay_sup_con', type=float, default=1e-4)        # 0.0005 #1e-4
    # parser.add_argument('-cosine_sup_con', action="store_true")
    # parser.add_argument('-lr_decay_rate_sup_con', type=float, default=0.1)                                                                             # 128
    # parser.add_argument('-lr_decay_epochs_sup_con', type=str, default='100,200',
    #                     help='where to decay lr, can be a list')# True
    # parser.add_argument('-warm_sup_con', action="store_true")
    # parser.add_argument('-margin_sup_con', type=float, default=0)    
    # parser.add_argument('-proto_init_sup_con', action="store_true")
    # parser.add_argument('-mixup_sup_con', action="store_true")
    # parser.add_argument('-train_base_in_novel', action="store_true")
    # parser.add_argument('-train_novel_in_joint', action="store_true")
    # parser.add_argument('-novel_high_layers', action="store_true")
    # parser.add_argument('-reserve_method', type=str, default='dist',
    #                     choices=['dist', 'angle', 'mixup', "hemisphere"])
    # parser.add_argument('-skip_encode_norm', action='store_true', help='')
    # parser.add_argument('-skip_sup_con_head', action='store_true', help='')
    # parser.add_argument('-skip_nonlinearity', action='store_true', help='')
    # parser.add_argument('-skip_novel', action='store_true', help='')
    # parser.add_argument('-new_head_hidden_dim', type=int, default=-1)      
    # parser.add_argument('-incremental_on_supcon_head', action='store_true', help='')
    # parser.add_argument('-fine_tune_backbone_joint', action='store_true', help='')
    
    # parser.add_argument('-nesterov_new', action='store_true', help='')
    
    # parser.add_argument('-joint_loss', type=str, default='ce_even',
    #                     choices=['ce_even', 'ce_inter', "ce", "ce_weighted", "tsc"])
    # parser.add_argument('-proto_method', type=str, help='Method used for prototype calculation', default="mean", choices=["mean", "gm"])
    # parser.add_argument('-hm_patience', type=int, default=15)
    # parser.add_argument('-compute_hardnegative', action="store_true")

    # # about dataset and network
    # parser.add_argument('-sweep_json', type=str, default="sweep_config/dino_sweep.json")
    # parser.add_argument('-freeze_backbone', action="store_true", help="Backbone of the resnet model is frozen")
    # parser.add_argument('-skip_wandb', action="store_true", help="Skip syncing with wandb")

    # # about data augmentation
    # parser.add_argument('-color_jitter', action="store_true", help="Apply Color Jitter")
    # parser.add_argument('-data_aug', action="store_true", help="Perform 2 image data augmentation")
    # parser.add_argument('-rand_aug', action="store_true", help="Perform random augmentation on CUB200")  

    
    # parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70]) # Cub200 -> [60, 120, 200]
    # parser.add_argument('-step', type=int, default=40)

    
    # parser.add_argument('-T_0_den', type=int, default=2, help="number of iterations for the first restart")
    # parser.add_argument('-eta_min', type=float, default=1e-2, help="Minimum learning rate when using cosine restart scheduler")
    # parser.add_argument('-joint_loading_method', type=str, default='old',
    #                     choices=['old', 'new', 'online'])

    # parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    # parser.add_argument('-label_smoothing', type=float, default=0.0)
    # parser.add_argument('-label_smoothing_novel', type=float, default=0.0)
    # parser.add_argument('-label_smoothing_joint', type=float, default=0.0)

    

    # parser.add_argument('-new_mode', type=str, default='ft_dot', # CEC -> avg_cos
    #                     choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_l2', 'dot']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # # for episode learning
    # parser.add_argument('-train_episode', type=int, default=50)
    # parser.add_argument('-episode_shot', type=int, default=1)
    # parser.add_argument('-episode_way', type=int, default=15)
    # parser.add_argument('-episode_query', type=int, default=15)
    # # parser.add_argument('-validation_metric', type=str, default='hm', choices=["hm", "loss", "acc", "none"])
    # # parser.add_argument('-validation_metric_novel', type=str, default='none', choices=["hm", "loss", "acc", "none"])
    # # parser.add_argument('-validation_metric_joint', type=str, default='none', choices=["hm", "loss", "acc", "none"])
    # parser.add_argument('-schedule_joint', action='store_true', help='Schedule the learning rate using cosine scheduler')
    # parser.add_argument('-rand_aug_joint', action='store_true', help='Schedule the learning rate using cosine scheduler')
    # parser.add_argument('-rand_aug_novel', action='store_true', help='Schedule the learning rate using cosine scheduler')
    


    # # for cec
    # parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    # parser.add_argument('-low_shot', type=int, default=1)
    # parser.add_argument('-low_way', type=int, default=15)
    # parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # # Reserve vector
    # parser.add_argument('-apply_bnce', action='store_true')
    # parser.add_argument('-use_ncfscil', action='store_true')
    # parser.add_argument('-instance_mixup', action='store_true')
    # parser.add_argument('-instance_mixup_joint', action='store_true')
    # parser.add_argument('-strong_transform', action='store_true')
    # parser.add_argument('-criterion', type=str, default='xent', # CEC -> ft_cos
    #                     choices=["xent", "cosine", "xent+cosine", "cosine-squared", "none"]) # ft_dot means using linear classifier, ft_cos means using cosine classifier

                        
    # parser.add_argument('-radial_label_smoothing', type=float, default=0)   
    # parser.add_argument('-xent_weight', type=float, default=0.1)   
    # parser.add_argument('-kp_lam', type=float, default=0.0)   
    # parser.add_argument('-lam_at', type=float, default=0)   
    # # parser.add_argument('-joint_supcon', action='store_true')
    # parser.add_argument('-testing_freq', type=float, default=5)
    # # parser.add_argument('-online_assignment', action='store_true')
    # # parser.add_argument("-assignment_mode_base", type=str, default="max",
    # #                     choices=["min", "random", "max"]) 
    # # parser.add_argument("-assignment_mode_novel", type=str, default="max",
    # #                     choices=["min", "random", "max", "cosine_penalty"])
    # # parser.add_argument('-assign_flip', action="store_true", help="flip assignment mode between min and max every session")   
    # # parser.add_argument('-reserve_init', type=str, default='randn', # CEC -> ft_cos
    # #                     choices=["randn", "proto"]) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    
    # parser.add_argument('-novel_bias', action="store_true")   
    # parser.add_argument('-queue_pull', action="store_true")

    
    # parser.add_argument('-warmup_epochs_base', type=int, default=3)
    # parser.add_argument('-warmup_epochs_inc', type=int, default=10)
    # parser.add_argument('-warmup_epochs_simplex', type=int, default=0)

    # # parser.add_argument('-assign_similarity_metric', type=str, default='cos', # CEC -> ft_cos
    # #                     choices=["cos", "euclidean", "mahalanobis", "cos_odd_inv"]) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    # parser.add_argument('-base_target_sampling', action="store_true")
    # parser.add_argument('-target_sampling', action="store_true") 
    # parser.add_argument('-supcon_views', type=int, default=2)
    # parser.add_argument('-supcon_even', action="store_true") 
    # parser.add_argument('-reserve_vector_count', type=int, default=-1)
    # parser.add_argument('-proj_output_dim', type=int, default=-1)
    # parser.add_argument('-proj_hidden_dim', type=int, default=2048)
    # parser.add_argument('-init_proj_random', action="store_true") 
    # parser.add_argument('-skip_base_ft', action="store_true") 
    # parser.add_argument('-append_hard_positives', action="store_true") 

    
    # # parser.add_argument('-mix_lam', type=float, default=0)
    

    # parser.add_argument("-pull_loss_mode", type=str, default="default",
    #                     choices=["default"])
    # parser.add_argument('-cos_b_margin', type=float, default=0) 
    # parser.add_argument('-cos_n_margin', type=float, default=0)

    # parser.add_argument('-dist_lam', type=float, default=0.5)
    # parser.add_argument('-proj_ema_update', action="store_true") 
    # parser.add_argument('-proj_ema_mode', type=str, default="prev", 
    #                     choices=["prev", "all", "base"]) 
    # parser.add_argument('-proj_ema_beta', type=float, default=0.9999) 
    # parser.add_argument('-proj_ema_update_every', type=int, default=10) 


    # # Extra Experiments
    # parser.add_argument('-heavy_inter_aug', action="store_true")  


    # parser.add_argument('-skip_perturbation', action="store_true")
    # parser.add_argument('-skip_prep', action="store_true")
    # parser.add_argument('-skip_orth', action="store_true")

    # parser.add_argument('-spread_aware', action="store_true", help="Pull the class features in the direction of the target offset by the original offset to retain geometry")

    # parser.add_argument('-class_aug_base', action="store_true", help="Class augmentation using mixup during base session")
    # parser.add_argument('-class_aug_inc', action="store_true", help="Class augmentation using mixup during incremental session")
    # parser.add_argument('-skip_perturbation_base', action="store_true", help="Class augmentation using mixup during incremental session")
    # parser.add_argument('-skip_perturbation_inc', action="store_true", help="Class augmentation using mixup during incremental session")

    # parser.add_argument('-remove_pert_numerator', action="store_true", help="Remove perturbations from numerator")
    # parser.add_argument('-remove_curr_features', action="store_true", help="Removes current task features from sup con loss")
    
    # 
    # # parser.add_argument('-apply_tbnce', action="store_true", help="Apply tbnce in all sessions")

    # parser.add_argument('-pull_delay', type=float, default=0)
    # parser.add_argument('-novel_bias_schedule', type=int, default=-1)
    
    # # parser.add_argument("-perturb_mode", type=str, default="inc-curr-base",
    # #                     choices=["inc-curr-base", "all", "inc+curr-base", "inc-curr+base", "all-ua", "inc+curr-base-ua"])

    # # parser.add_argument("-perturb_dist", type=str, default="uniform",
    #                     # choices=["uniform", "gaussian"])

    # # parser.add_argument("-perturb_mode_base", type=str, default="inc",
    # #                     choices=["all", "inc"])
    # # parser.add_argument("-perturb_mode_inc", type=str, default="inc-curr-base",
    # #                     choices=["inc-curr-base", "all", "inc+curr-base", "inc-curr+base"])

    

    return parser

if __name__ == '__main__':
    # Parse Arguments
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    
    # Trainer initialization
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()