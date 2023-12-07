import sys
import os
sys.path.append(os.path.dirname(__file__))

from base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from helper import *
from supcon import *
from utils import *
from dataloader.data_utils import *

from gaussian_utils import *

import wandb
from torch.utils.tensorboard import SummaryWriter

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.writer = SummaryWriter(os.path.join(self.args.save_path, "logs"))

        self.model = MYNET(self.args, mode=self.args.base_mode, writer = self.writer)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.load_strict = True
        if self.args.model_dir is not None and not args.resume:
            # Load the solo-learn model here
            if self.args.model_dir.endswith(".ckpt"):
                solo_dict = torch.load(self.args.model_dir)["state_dict"]
                self.best_model_dict = {}
                for k,v in solo_dict.items():
                    if "backbone" in k:
                        self.best_model_dict[k.replace("backbone", "module.encoder")] = v
                    if "projector" in k:
                        self.best_model_dict["module." + k] = v

                self.load_strict = False
            elif self.args.model_dir.endswith(".pth.tar"): # For MOCO_supcon
                moco_dict = torch.load(self.args.model_dir)["state_dict"]
                # Select the query encoder only
                self.best_model_dict = {}
                for k,v in moco_dict.items():
                    if "encoder_q" in k:  # Selecting only the query encoder
                        self.best_model_dict[k.replace("encoder_q", "encoder")] = v
                self.load_strict = False
            else:
                print('Loading init parameters from: %s' % self.args.model_dir)
                ld = torch.load(self.args.model_dir)
                if "params" not in ld.keys():
                    # ALICE model. replace encoder with module.encoder
                    alice_dict = ld['state_dict']
                    self.best_model_dict = {}
                    for k,v in alice_dict.items():
                        if "backbone" in k:
                            self.best_model_dict[k.replace("backbone", "module.encoder")] = v
                    self.load_strict = False
                else:
                    og_dict = ld['params']
                    self.best_model_dict = {}
                    for k,v in og_dict.items():
                        if "sup_con_head" in k:
                            self.best_model_dict[k.replace("sup_con_head", "module.projector")] = v
                        if "encoder" in k:
                            self.best_model_dict[k] = v
                        
                    self.load_strict = False
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        # Satrting with novel session directly
        args.start_session = 1

        self.strong_transform = get_strong_transform(args)

    def get_optimizer_base(self, lr = None, freeze_backbone = False):
        if lr is None:
            lr = self.args.lr_base

        if not freeze_backbone:
            params = self.model.parameters()
        else:
            params = self.model.module.fc.parameters()

        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler        
    
    def get_optimizer_sup_con(self):
        lr = self.args.lr_sup_con
        params = self.model.parameters()

        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr, momentum=self.args.momentum_sup_con, nesterov=True, weight_decay=self.args.decay_sup_con)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr, weight_decay=self.args.decay_sup_con)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler  

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        gaussian_prototypes = None
        
        if args.sup_con_pretrain:
            base_set, base_trainloader, base_testloader = get_base_dataloader(args)
            if args.model_dir is not None:
                save_model_dir = os.path.join(args.save_path, 'session0_max_acc.pth')

                can_resume = args.resume and os.path.exists(save_model_dir)
                if can_resume:
                    # Resuming the model with the reserved vectors + trained base classifier
                    self.best_model_dict = torch.load(save_model_dir)["params"]

                    if not self.args.online_assignment:
                        # Base resume would mean that the reserved hyperspherical prototypes
                        # would be reduced to just a n_inc base classes
                        # self.model.module.fc.rv = self.model.module.fc.rv[:self.model.module.fc.n_inc_classes]
                        self.model.module.fc.rv = self.model.module.fc.rv[self.args.base_class:]

                    # Load model
                    self.model.load_state_dict(self.best_model_dict, strict = True)

                    # Compute validation score
                    # out = test(self.model, base_testloader, 0, args, 0, base_sess = True)
                    out = test_fixed(self.model, base_testloader, 0, args, 0)
                    best_va = out[1]
                else:
                    self.model.load_state_dict(self.best_model_dict, strict = self.load_strict)
                    
                    # Get mean prototypes from the projection head
                    best_prototypes = get_base_fc(base_set, base_testloader.dataset.transform, self.model, args)

                    # Compute validation score
                    _, best_va, _, _, _, _, _ = self.model.module.test_fc(best_prototypes, base_testloader, 0, 0)

                    # Assign projector prototypes to the base classifier
                    print("===Assigned Base Prototypes===")
                    self.model.module.fc.classifiers[0].weight.data = best_prototypes

                    print("===Compute Reserved Vectors===")
                    self.model.module.fc.find_reseverve_vectors_all()

                    # Assign classifier to the best prototypes
                    self.model.module.fc.assign_base_classifier(best_prototypes)

                    print("===Fine tuning projection head for base classifier===")
                    # Fine tune projection unit, update the fc.classifiers[0] to the new prototypes
                    # best_va = self.model.module.update_fc_ft_base(base_trainloader, base_testloader)

                    # Get sup con dataloader for base session
                    _, sup_trainloader, _ = get_supcon_dataloader(self.args)
                    best_va = self.model.module.update_fc_ft_base(sup_trainloader, base_testloader)

                    # Save model and the reserve 
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
            else:
                raise NotImplementedError()

            print(f"===Final Accuracy with Assigned Base Prototypes: {best_va*100:.3f}")
            self.trlog['max_acc'][0] = float('%.3f' % (best_va * 100))

        if args.end_session == -1:
            args.end_session = args.sessions

        print("===Starting Incremental Sessions===")
        for session in range(args.start_session, args.end_session):
            if gaussian_prototypes is None:
                print("===Gaussian Modelling===")
                base_set, base_trainloader, base_testloader = get_base_dataloader(args)
                gaussian_obj = get_gaussian_for_session(None, None, self.model, args, 0, use_vector_variance=args.use_vector_variance, views=self.args.views_gaussian, trainloader=base_trainloader)

            # Load data for this session
            train_set, trainloader, testloader = self.get_dataloader(session)

            # Load base model/previous incremental session model
            self.model.load_state_dict(self.best_model_dict, strict = True)

            # Initialising projector ema for dist loss
            if self.args.proj_ema_update and self.model.module.projector_ema is None:
                self.model.module.init_proj_ema()

            print("training session: [%d]" % session)
            
            self.model.eval()

            # Update the correct novel fc using the linear sum assignment matching. Matching each novel class to the closesnt reserve vector.
            self.model.module.update_fc(trainloader, testloader, np.unique(train_set.targets), session)

            # Update projection head
            joint_set, jointloader = get_supcon_new_dataloader(args, session)
            self.model.module.update_fc_ft_joint_supcon(jointloader, testloader, np.unique(joint_set.targets), session, gaussian_obj)

            # Update Gaussian obj
            gaussian_obj_new = get_gaussian_for_session(None, None, self.model, args, session, use_vector_variance=args.use_vector_variance, views=self.args.views_gaussian, trainloader = trainloader)
            gaussian_obj["prototypes"] = np.vstack((gaussian_obj["prototypes"], gaussian_obj_new["prototypes"]))
            gaussian_obj["prototypeLabels"] = np.hstack((gaussian_obj["prototypeLabels"], gaussian_obj_new["prototypeLabels"]))
            gaussian_obj["classVariances"] = np.vstack((gaussian_obj["classVariances"], gaussian_obj_new["classVariances"]))

            tsl, tsa, tsaNovel, tsaBase, vaSession = test_fixed(self.model, testloader, 0, args, session)

            # Save Accuracies and Means
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_novel_acc'][session] = float('%.3f' % (tsaNovel * 100))
            self.trlog['max_base_acc'][session] = float('%.3f' % (tsaBase * 100))
            self.trlog["max_hm"][session] = float('%.3f' % (hm(tsaBase, tsaNovel) * 100))
            self.trlog["max_am"][session] = float('%.3f' % (am(tsaBase, tsaNovel) * 100))

            # Save the best model
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('Saving model to :%s' % save_model_dir)

            out_string = 'Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}, am {:.3f}'\
                .format(
                    session, 
                    self.trlog['max_acc'][session], 
                    self.trlog['max_novel_acc'][session], 
                    self.trlog['max_base_acc'][session], 
                    self.trlog["max_hm"][session],
                    self.trlog["max_am"][session]
                    )

            print(out_string)

            def print_sessional_acc(vaSession):
                output = "Session score=> "
                for ix, sess in enumerate(vaSession):
                    # print(f"Session {ix} Accuracy: {sess * 100:.3f}")
                    output += f" {ix}:{sess * 100:.3f},"
                output += "\n"
                print(output)

            print_sessional_acc(vaSession)

            result_list.append(out_string)

            self.writer.flush()
        
        # Remove the firsat dummy harmonic mean value
        del self.trlog['max_hm'][0]

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))

        result_list.append("Top 1 Accuracy: ")
        result_list.append(self.trlog['max_acc'])
        
        result_list.append("Harmonic Mean: ")
        result_list.append(self.trlog['max_hm'])

        result_list.append("Arithematic Mean: ")
        result_list.append(self.trlog['max_am'])

        result_list.append("Base Test Accuracy: ")
        result_list.append(self.trlog['max_base_acc'])

        result_list.append("Novel Test Accuracy: ")
        result_list.append(self.trlog['max_novel_acc'])

        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        result_list.append("Average Harmonic Mean Accuracy: ")
        result_list.append(average_harmonic_mean)

        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        result_list.append("Average Harmonic Mean Accuracy: ")
        result_list.append(average_harmonic_mean)

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
        print(f"pd: {performance_decay:.3f}")
        print(f"am: {self.trlog['max_am']}")
        print(f"base: {self.trlog['max_base_acc']}")
        print(f"novel: {self.trlog['max_novel_acc']}")

        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        self.writer.close()

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

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
        ensure_path(self.args.save_path)

        # Make sub folders for confusion matrix
        self.cm_path = os.path.join(self.args.save_path, "confusion_matrix") 
        if not os.path.exists(self.cm_path):
            os.mkdir(self.cm_path)

        return None