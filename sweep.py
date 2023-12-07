import numpy as np
import wandb
import argparse
import importlib
from utils import *
import os
from train import *
import json

# sweep_config = {
#     'method': 'random', #grid, random
#     'metric': {
#       'name': 'avg_hm', # average harmoic mean would optimize over the entire set
#       'goal': 'maximize'
#     },
#     'parameters': {
#         # 'epochs_base': {
#         #     'values': [100, 200, 300]
#         # },
#         # 'mixup_layer_choice':{
#         #     'values': ["1", "1,2", "1,2,3", "1,2,3,4"] # mixup hidden will be true
#         # },
#         # 'mixup_novel': {
#         #     'values': ["true", "false"]
#         # },
#         # 'mixup_joint': {
#         #     'values': ["true", "false"]
#         # },
#         # 'mixup_alpha': {
#         #     'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#         # },
#         # 'label_smoothing': {
#         #     # "distribution": "uniform",
#         #     # "min": 0.5, "max": 0.95
#         #     "values": [0.6, 0.8, 0.9, 0.95]
#         # }
#         ### == Dino search
#         'mixup_start_epoch': {
#             "distribution": "uniform",
#             "min": 10, "max": 100
#         },
#         # 's2m2':{
#         #     "values":["true", "false"]
#         # },
#         'teacher_scheduling':{
#             "values":["true", "false"]
#         },
#         'dino_loss_weight': {
#             "distribution": "uniform",
#             "min": 0.0001, 
#             "max": 0.1
#         },
#         'classifier_last_layer':{
#             "values":['linear', 'projection']
#         },
#         'classifier_hidden_dim':{
#             "values":[256, 512, 1024, 2048]
#         },
#         ### == Dino Loss Weight
#         # 'dino_loss_weight': {
#         #     "values":[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
#         # },
#     }
# }

# Intialise arguments
parser = get_command_line_parser()
common_args = parser.parse_args()
set_seed(common_args.seed)
common_args.num_gpu = set_gpu(common_args)
common_args.mixup_layer_choice = set_layers(common_args) # Setting layer choice for mixup
with open(common_args.sweep_json) as json_file:
    sweep_config = json.load(json_file)

def train():
    # Setup trainer
    trainer = importlib.import_module('models.%s.fscil_trainer' % (common_args.project)).FSCILTrainer(common_args)

    # Run the train function
    trainer.train()

if __name__ == '__main__':
    sweep_project = common_args.sweep_json.split(os.path.sep)[-1].split(".")[0]
    
    # Initialise a sweep
    sweep_id = wandb.sweep(sweep_config, entity="noorahmedds", project=sweep_project)

    # Once the agent is run each agent then has a train an associated config. And wandb init will take care of the rest
    wandb.agent(sweep_id, train)