import sys
import os
sys.path.append(os.path.dirname(__file__))

from base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

import helper
from supcon import *
import utils
import dataloader.data_utils as data_utils
from Network import ORCONET
import time

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()                                                                            # Setting logs and artefact paths.
        self.args = data_utils.set_up_datasets(self.args)                                               # Data wrapper inside the args object, passed throughout this file

        self.model = ORCONET(self.args, mode=self.args.base_mode)                                         # Initializing network
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))                        
        self.model = self.model.cuda()

        self.best_model_dict = {}
        if self.args.model_dir is not None:                                                             # Loading pretrained model, Note that for CUB we use an imagenet pretrained model            
            state_dict = torch.load(self.args.model_dir)["state_dict"]
            # Adapt keys to match network
            for k,v in state_dict.items():
                if "backbone" in k:
                    self.best_model_dict[k.replace("backbone", "module.encoder")] = v
                if "projector" in k:
                    self.best_model_dict["module." + k] = v

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = data_utils.get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = data_utils.get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train_phase2(self):
        """
            Base Alignment Phase: We aim to align the base dataset $D^0$ to the pseudo-targets through our OrCo loss. 
        """
        base_set, _, base_testloader = self.get_dataloader(0)
        save_model_dir = os.path.join(self.args.save_path, 'session0_max_acc.pth')
        
        if len(self.best_model_dict):
            self.model.load_state_dict(self.best_model_dict, strict=False)

        # Compute the Mean Prototypes (Indicated as \mu_j in the paper) from the projection head
        best_prototypes = helper.get_base_prototypes(base_set, base_testloader.dataset.transform, self.model, self.args)

        print("===Compute Pseudo-Targets and Class Assignment===")
        self.model.module.fc.find_reseverve_vectors_all()

        print("===[Phase-2] Started!===")
        self.model.module.fc.assign_base_classifier(best_prototypes)        # Assign classes to the optimal pseudo target
        _, sup_trainloader, _ = data_utils.get_supcon_dataloader(self.args)
        self.model.module.update_base(sup_trainloader, base_testloader)

        # Save Phase-1 model
        torch.save(dict(params=self.model.state_dict()), save_model_dir)
        self.best_model_dict = deepcopy(self.model.state_dict())

        # Compute Phase-2 Accuracies
        out = helper.test(self.model, base_testloader, 0, self.args, 0)
        best_va = out[0]

        # Log the Phase-2 Accuracies
        print(f"[Phase-2] Accuracy: {best_va*100:.3f}")
        self.trlog['max_acc'][0] = float('%.3f' % (best_va * 100))

    def train(self):
        t_start_time = time.time()

        # Train Stats
        result_list = [self.args]
        
        # Base Alignment (Phase 2)
        self.train_phase2()

        # Set the projector in reset setting
        self.model.module.set_projector()

        # Few-Shot Alignment (Phase 3)
        # for session in range(1, self.args.sessions):
        for session in range(1, 2):
            # Load base model/previous incremental session model
            self.model.load_state_dict(self.best_model_dict, strict = True)

            # Resetting the projection head to base             
            self.model.module.reset_projector()

            # Load data for this session
            train_set, trainloader, testloader = self.get_dataloader(session)

            print("\n\n===[Phase-3][Session-%d] Started!===" % session)
            self.model.eval() # Following CEC

            # Assignment
            self.model.module.update_targets(trainloader, testloader, np.unique(train_set.targets), session)

            # Alignment
            _, jointloader = data_utils.get_supcon_joint_dataloader(self.args, session)
            self.model.module.update_incremental(jointloader, session)

            # Compute scores
            tsa, novel_cw, base_cw = helper.test(self.model, testloader, 0, self.args, session)

            # Save Accuracies and Means
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_novel_acc'][session] = float('%.3f' % (novel_cw * 100))
            self.trlog['max_base_acc'][session] = float('%.3f' % (base_cw * 100))
            self.trlog["max_hm"][session] = float('%.3f' % (utils.hm(base_cw, novel_cw) * 100))

            # Save the final model
            save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('===[Phase-3][Session-%d] Saving model to :%s===' % (session, save_model_dir))

            out_string = 'Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}'\
                .format(
                    session, 
                    self.trlog['max_acc'][session], 
                    self.trlog['max_novel_acc'][session], 
                    self.trlog['max_base_acc'][session], 
                    self.trlog["max_hm"][session]
                    )
            print(out_string)

            result_list.append(out_string)
        
        self.exit_log(result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Total time used %.3f mins' % total_time)

    def exit_log(self, result_list):
        # Remove the firsat dummy harmonic mean value
        del self.trlog['max_hm'][0]
        del self.trlog['max_hm_cw'][0]

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))

        result_list.append("Top 1 Accuracy: ")
        result_list.append(self.trlog['max_acc'])
        
        result_list.append("Harmonic Mean: ")
        result_list.append(self.trlog['max_hm'])

        result_list.append("Base Test Accuracy: ")
        result_list.append(self.trlog['max_base_acc'])

        result_list.append("Novel Test Accuracy: ")
        result_list.append(self.trlog['max_novel_acc'])

        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        result_list.append("Average Harmonic Mean Accuracy: ")
        result_list.append(average_harmonic_mean)

        average_acc = np.array(self.trlog['max_acc']).mean()
        result_list.append("Average Accuracy: ")
        result_list.append(average_acc)

        performance_decay = self.trlog['max_acc'][0] - self.trlog['max_acc'][-1]
        result_list.append("Performance Decay: ")
        result_list.append(performance_decay)

        print(f"\n\nacc: {self.trlog['max_acc']}")
        print(f"avg_acc: {average_acc:.3f}")
        print(f"hm: {self.trlog['max_hm']}")
        print(f"avg_hm: {average_harmonic_mean:.3f}")
        print(f"pd: {performance_decay:.3f}")
        print(f"base: {self.trlog['max_base_acc']}")
        print(f"novel: {self.trlog['max_novel_acc']}")    
        utils.save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), result_list)


    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset                      
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        # Appending a user defined prefix to the folder.
        if self.args.save_path_prefix:
            self.args.save_path = self.args.save_path + self.args.save_path_prefix + "_" 

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        utils.ensure_path(self.args.save_path)