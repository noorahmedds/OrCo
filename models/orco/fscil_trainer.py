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
# from dataloader.data_utils import *
import dataloader.data_utils as data_utils
from Network import MYNET
import time

from gaussian_utils import *

import wandb

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()                                                                            # Setting logs and artefact paths.
        self.args = data_utils.set_up_datasets(self.args)                                               # Data wrapper inside the args object, passed throughout this file

        self.model = MYNET(self.args, mode=self.args.base_mode)                                         # Initializing network
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))                        
        self.model = self.model.cuda()

        if self.args.model_dir is not None:                                                             # Loading pretrained model            
            state_dict = torch.load(self.args.model_dir)["state_dict"]
            self.best_model_dict = {}
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
        base_set, base_trainloader, base_testloader = self.get_dataloader(0)
        save_model_dir = os.path.join(self.args.save_path, 'session0_max_acc.pth')
        self.model.load_state_dict(self.best_model_dict, strict=False)

        # Get mean prototypes from the projection head
        best_prototypes = helper.get_base_fc(base_set, base_testloader.dataset.transform, self.model, self.args)
        # if args.model_dir is not None:  # Loading Pretrained model (Minet, CIFAR)
            # # Compute prototypical validation score ahead of training
            # best_va = self.model.module.test_fc(best_prototypes, base_testloader, 0, 0)[1]

            # # Assign projector prototypes to the base classifier
            # print("===Assigned Base Prototypes===")
            # self.model.module.fc.classifiers[0].weight.data = best_prototypes

        print("===Compute Pseudo-Targets and Class Assignment===")
        self.model.module.fc.find_reseverve_vectors_all()
        self.model.module.fc.assign_base_classifier(best_prototypes)        # Assign classifier to the best prototypes

        print("===[Phase-2] Aligning Base Classes to Pseudo Targets===")
        _, sup_trainloader, _ = data_utils.get_supcon_dataloader(self.args)
        self.model.module.update_fc_ft_base(sup_trainloader, base_testloader)

        # Save Phase-1 model
        torch.save(dict(params=self.model.state_dict()), save_model_dir)
        self.best_model_dict = deepcopy(self.model.state_dict())

        # Compute Base Accuracy
        out = helper.test(self.model, base_testloader, 0, self.args, 0)
        best_va = out[1]

        # Log
        print(f"[Phase-2] Accuracy: {best_va*100:.3f}")
        self.trlog['max_acc'][0] = float('%.3f' % (best_va * 100))

    def train(self):
        args = self.args
        t_start_time = time.time()

        # Train Stats
        result_list = [args]
        
        # Base Alignment (Phase 2)
        self.train_phase2()

        # Set the projector in reset setting
        # TODO: Check if equivalent
        # self.best_projector = deepcopy(self.model.module.projector.state_dict())
        self.model.module.set_projector()

        # Few-Shot Alignment (Phase 3)
        for session in range(1, self.args.sessions):
            # Load base model/previous incremental session model
            self.model.load_state_dict(self.best_model_dict, strict = True)

            if self.args.init_sess_w_base_proj:             # Resetting the projection head
                self.model.module.reset_projector()

            # Load data for this session
            train_set, trainloader, testloader = self.get_dataloader(session)

            print("===[Phase-3][Session-%d] started!" % session)
            self.model.eval() # Following CEC          

            print("===[Phase-3][Session-%d] Assignment" % session)
            self.model.module.update_fc(trainloader, testloader, np.unique(train_set.targets), session)

            print("===[Phase-3][Session-%d] Alignment" % session)
            joint_set, jointloader = data_utils.get_supcon_joint_dataloader(self.args, session, self.model.module.path2conf)
            self.model.module.update_fc_ft_joint_supcon(jointloader, testloader, np.unique(joint_set.targets), session)

            # Compute scores
            tsl, tsa, tsaNovel, tsaBase, vaSession, cw_acc, novel_cw, base_cw, cos_sims, fpr = helper.test(self.model, testloader, 0, self.args, session)

            # Save Accuracies and Means
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_novel_acc'][session] = float('%.3f' % (tsaNovel * 100))
            self.trlog['max_base_acc'][session] = float('%.3f' % (tsaBase * 100))
            self.trlog["max_hm"][session] = float('%.3f' % (hm(tsaBase, tsaNovel) * 100))
            self.trlog["max_hm_cw"][session] = float('%.3f' % (hm(base_cw, novel_cw) * 100))

            # Save the best model
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('===[Phase-3][Session-%d] Saving model to :%s' % (session, save_model_dir))

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
        
        result_list.append("Harmonic Mean (Class-Wise Accuracy): ")
        result_list.append(self.trlog['max_hm_cw'])

        result_list.append("Base Test Accuracy: ")
        result_list.append(self.trlog['max_base_acc'])

        result_list.append("Novel Test Accuracy: ")
        result_list.append(self.trlog['max_novel_acc'])

        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        result_list.append("Average Harmonic Mean Accuracy: ")
        result_list.append(average_harmonic_mean)

        average_harmonic_mean_cw = np.array(self.trlog['max_hm_cw']).mean()
        result_list.append("Average Harmonic Mean Accuracy Class-Wise: ")
        result_list.append(average_harmonic_mean_cw)

        average_acc = np.array(self.trlog['max_acc']).mean()
        result_list.append("Average Accuracy: ")
        result_list.append(average_acc)

        performance_decay = self.trlog['max_acc'][0] - self.trlog['max_acc'][-1]
        result_list.append("Performance Decay: ")
        result_list.append(performance_decay)

        print(f"acc: {self.trlog['max_acc']}")
        print(f"avg_acc: {average_acc:.3f}")
        print(f"hm: {self.trlog['max_hm']}")
        print(f"avg_hm: {average_harmonic_mean:.3f}")
        print(f"hm_cw: {self.trlog['max_hm_cw']}")
        print(f"avg_hm_cw: {average_harmonic_mean_cw:.3f}")
        print(f"pd: {performance_decay:.3f}")
        print(f"base: {self.trlog['max_base_acc']}")
        print(f"novel: {self.trlog['max_novel_acc']}")
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])        
        utils.save_list_to_txt(os.path.join(self.args.save_path, 'results.txt'), result_list)

    def visualise_encoding(self, session):
        # visualise
        _, _, testloader = self.get_dataloader(session)
        tqdm_gen = tqdm(testloader)
        encodings = []
        labels = []
        self.model = self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = batch
                data = data.cuda()
                enc = self.model.module.encode(data)
                encodings.append(enc)
                labels.append(test_label)
            
        # Now we concatenate all encoding
        encodings = torch.cat(encodings).cpu()
        labels = torch.cat(labels).unsqueeze(-1)
        comb = torch.hstack([encodings, labels])
        # filter only 10 classes
        # comb = comb[comb[:, -1] >= 100]

        columns = [f"D{i}" for i in range(512)]
        columns.append("gt")
        
        wandb.log({"embeddings": wandb.Table(columns = columns, data = comb.numpy().tolist())})

    def sessional_test(self):
        # For each session compute prototypes, replace classifiers and then compute the accuracies as usual but without training anything
        args = self.args
        base_set, base_trainloader, base_testloader = get_base_dataloader(args)

        self.model.load_state_dict(self.best_model_dict, strict = self.load_strict)

        # Get mean prototypes from the projection head
        best_prototypes = get_base_fc(base_set, base_testloader.dataset.transform, self.model, args, mode="backbone")

        _, best_va, _, _, _, _, _ = self.model.module.test_backbone(best_prototypes, base_testloader, 0, 0)

        print(f"===Final Accuracy for base session: {best_va*100:.3f}")
        self.trlog['max_acc'][0] = float('%.3f' % (best_va * 100))

        if args.end_session == -1:
            args.end_session = args.sessions

        all_prototypes = [best_prototypes]
        for session in range(1, args.end_session):
            # Load data for this session
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.eval()

            # Update the correct novel fc using the linear sum assignment matching. Matching each novel class to the closesnt reserve vector.
            new_prototypes = get_new_fc(train_set, base_testloader.dataset.transform, self.model, args, np.unique(train_set.targets), views=10, mode="backbone")
            all_prototypes.append(new_prototypes)

            tsl, tsa, tsaNovel, tsaBase, tsHm, tsAm, _ = self.model.module.test_backbone(torch.cat(all_prototypes, axis = 0), testloader, 0, 0)

            # Save Accuracies and Means
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_novel_acc'][session] = float('%.3f' % (tsaNovel * 100))
            self.trlog['max_base_acc'][session] = float('%.3f' % (tsaBase * 100))
            self.trlog["max_hm"][session] = float('%.3f' % (hm(tsaBase, tsaNovel) * 100))
            self.trlog["max_am"][session] = float('%.3f' % (am(tsaBase, tsaNovel) * 100))

            out_string = 'Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}, am {:.3f}\n'\
                .format(
                    session, 
                    self.trlog['max_acc'][session], 
                    self.trlog['max_novel_acc'][session], 
                    self.trlog['max_base_acc'][session], 
                    self.trlog["max_hm"][session],
                    self.trlog["max_am"][session]
                    )
            print(out_string)

        del self.trlog['max_hm'][0]
        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        average_acc = np.array(self.trlog['max_acc']).mean()
        performance_decay = self.trlog['max_acc'][0] - self.trlog['max_acc'][-1]

        print(f"acc: {self.trlog['max_acc']}")
        print(f"avg_acc: {average_acc:.3f}")
        print(f"hm: {self.trlog['max_hm']}")
        print(f"avg_hm: {average_harmonic_mean:.3f}")
        print(f"pd: {performance_decay:.3f}")
        print(f"am: {self.trlog['max_am']}")
        print(f"base: {self.trlog['max_base_acc']}")
        print(f"novel: {self.trlog['max_novel_acc']}")

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset                      
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        if self.args.save_path_prefix:
            self.args.save_path = self.args.save_path + self.args.save_path_prefix + "_"        # Appending a user defined prefix to the folder

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftnEpoch_%d-jEpoch_%d' % (
                self.args.lr_new, self.args.epochs_novel, self.args.epochs_joint)

        self.args.save_path = self.args.save_path + '-opt_%s' % (self.args.optimizer)
        self.args.save_path = self.args.save_path + '-vm_%s' % (self.args.validation_metric)
        self.args.save_path = self.args.save_path + '-ls_%.2f' % (self.args.label_smoothing)
        self.args.save_path = self.args.save_path + '-ma_%.2f' % (self.args.mixup_alpha)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        utils.ensure_path(self.args.save_path)

        # Make sub folders for confusion matrix
        self.cm_path = os.path.join(self.args.save_path, "confusion_matrix") 
        if not os.path.exists(self.cm_path):
            os.mkdir(self.cm_path)

        return None
