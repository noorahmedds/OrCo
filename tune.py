import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = '../../datasets/'
PROJECT='base'
SAVE_PATH_PREFIX = ""

from ray import air, tune
import ray

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about hyper parameter tuning
    parser.add_argument('-tune_hp', action="store_true", help="Tune hyper parameters")

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-freeze_backbone', action="store_true", help="Backbone of the resnet model is frozen")
    parser.add_argument('-save_path_prefix', "-prefix", type=str, default=SAVE_PATH_PREFIX)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1) # CUB200 Anna -> 5e-3
    parser.add_argument('-optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70]) # Cub200 -> [60, 120, 200]
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-decay_new', type=float, default=0) # CUB200 Anna -> 1e-4
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)  # CUB 200 Anna -> 64
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    # for cec
    parser.add_argument('-lrg', type=float, default=0.1) #lr for graph attention network
    parser.add_argument('-low_shot', type=int, default=1)
    parser.add_argument('-low_way', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    return parser

def update_args(args, config):
    for k,v in config.items():
        setattr(args, k, v)
    return args

def trainer(config):
    ss = config["search_space"]
    print(ss)
    result = ss["lr_new"] * ss["momentum"] * ss["decay_new"]
    print(torch.cuda.is_available)
    # tune.report({"fin_harmonic_mean": result})  # Report to Tune
    tune.report(fin_harmonic_mean = result)  # Report to Tune

def hp_search(config):
    args = update_args(config["common_args"], config["search_space"])
    os.chdir("/BS/fscil/work/code/CEC-CVPR2021/")       # As found here: https://github.com/ray-project/ray/issues/9571
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    search_space = {
        "common_args": args, 
        "search_space":{
            "lr_new": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "momentum": tune.uniform(0.5, 0.9),
            "optimizer": tune.choice(["sgd", "adam"]),
            "decay_new": tune.uniform(1e-4, 0),
            "epochs_new": tune.uniform(25, 200)
        }
    }

    ray.init(num_gpus=args.num_gpu)    # Define max cpu and num gpus to be used

    # https://mengliuz.medium.com/hyperparameter-tuning-for-deep-learning-models-with-the-ray-simple-pytorch-example-da7b17e3505
    results = tune.run(
            # trainer,   # the core training/testing of your model
            hp_search,   # the core training/testing of your model
            local_dir=os.getcwd(), # for saving the log files
            name="hp_tune_logs", # name for the result directory
            metric='fin_harmonic_mean',
            mode='max',
            max_concurrent_trials=2,
            resources_per_trial={
                "cpu": 4,
                "gpu": 1
            },
            num_samples=50, # 50 trials
            config=search_space
            )

    print("Best config is:", results.best_config)
    ray.shutdown()

    # dfs = {result.log_dir: result.metrics_dataframe for result in results}
    # [d.mean_accuracy.plot() for d in dfs.values()]