from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *

import wandb
from torch.utils.tensorboard import SummaryWriter

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        try:
            if not self.args.skip_wandb:
                wandb.init(project=self.args.project, sync_tensorboard = True)
                
                # HP search
                wandb_configs = wandb.config

                # Setting up sweep parameters. 
                if len(wandb_configs):
                    self.args = set_up_sweep_args(self.args, wandb_configs)
                    pprint(vars(self.args))
                    
        except Exception as e:
            print(e)
            print("**WANDB Could not be initialised**")

        self.set_save_path()
        self.args = set_layers(self.args) # Setting layer choice for mixup
        self.args = set_up_datasets(self.args)

        self.writer = SummaryWriter(os.path.join(self.args.save_path, "logs"))

        self.model = MYNET(self.args, mode=self.args.base_mode, writer = self.writer)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                        weight_decay=self.args.decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr_base, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args, debug=False)
        elif session == -1:
            trainset, trainloader, testloader = get_rot_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def s2m2(self, session = -1):
        # Rotate each input image and learn a 4 way classifier on top that learns the rotation. Then apply manifold mixup
        args = self.args
        
        self.model.module.mode = "ft_rot"
        optimizer, scheduler = self.get_optimizer_base()
        
        max_acc = 0
        max_acc_epoch = 0

        train_set, trainloader, testloader = self.get_dataloader(session = -1)  
        for epoch in range(args.epochs_base):
            # train base sess
            tl, ta = base_rot_train(self.model, trainloader, optimizer, scheduler, epoch, args, session = session)
            
            # test model with all seen class
            tsl, tsa = rot_test(self.model, testloader, epoch, args, session)

            # save better model
            if (tsa * 100) >= max_acc:
                max_acc_epoch = epoch
                max_acc = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('********A better model is found!!**********')
                print('Saving model to :%s' % save_model_dir)
            # else:
            #     if session == 0:
            #         # If the fine tuning yields worse result than the max acc then we return the current model
            #         save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            #         torch.save(dict(params=self.model.state_dict()), save_model_dir)
            #         torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
            #         self.best_model_dict = deepcopy(self.model.state_dict())
            #         return

                
            print('best epoch {}, best test acc={:.3f}'.format(max_acc_epoch, max_acc))

            # Logging to tensorboard
            self.writer.add_scalars(f'Self Supervised Session/', {
                "train_accuracy":ta,
                "test_accuracy": tsa
            }, epoch)

            scheduler.step()

    def train(self):
        args = self.args
        t_start_time = time.time()

        # wandb.init(project=args.project, config=args, sync_tensorboard=True)

        # ==== Self supervised pretraining ====
        print("Pre training on rotation auxilliary task")
        # self.s2m2(session = -1)

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            # Load the model we learned from the rotation module
            self.model.load_state_dict(self.best_model_dict, strict = True)

            # Logging gradients
            # wandb.watch(self.model, log_freq=100)

            if session == 0:  # load base class train img label
                # self.model.module.mode = self.args.base_mode

                # print('new classes for this session:\n', np.unique(train_set.targets))
                # optimizer, scheduler = self.get_optimizer_base()

                # for epoch in range(args.epochs_base):
                #     start_time = time.time()
                #     # train base sess
                #     tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    
                #     # test model with all seen class
                #     tsl, tsa = (test(self.model, testloader, epoch, args, session, base_sess = True))[:2]

                #     # save better model
                #     if (tsa * 100) >= self.trlog['max_acc'][session]:
                #         self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                #         self.trlog['max_acc_epoch'] = epoch
                #         save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #         torch.save(dict(params=self.model.state_dict()), save_model_dir)
                #         torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                #         self.best_model_dict = deepcopy(self.model.state_dict())
                #         print('********A better model is found!!**********')
                #         print('Saving model to :%s' % save_model_dir)
                #     print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                #                                                        self.trlog['max_acc'][session]))

                #     # Logging to tensorboard
                #     self.writer.add_scalars(f'Base Session/', {
                #         "train_accuracy":ta,
                #         "test_accuracy": tsa
                #     }, epoch)

                #     self.trlog['train_loss'].append(tl)
                #     self.trlog['train_acc'].append(ta)
                #     self.trlog['test_loss'].append(tsl)
                #     self.trlog['test_acc'].append(tsa)
                #     lrc = scheduler.get_last_lr()[0]
                #     result_list.append(
                #         'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                #             epoch, lrc, tl, ta, tsl, tsa))
                #     print('This epoch takes %d seconds' % (time.time() - start_time),
                #           '\nstill need around %.2f mins to finish this session' % (
                #                   (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                #     scheduler.step()

                # # Alice Mixup: Remove the final fc layer with a fc of size num_class
                # # self.model = replace_mixup_fc(self.model, args)
                # # self.best_model_dict = replace_mixup_fc(self.best_model_dict, args)

                # result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                #     session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                # # ==== Fine tuning on the same task but with manifold mixup ====
                # # print("======Fine Tuning with Manifold Mixup======")
                # # self.s2m2(session = 0)

                # Pretraining with self supervised rotation training alongside class training
                self.s2m2(session = -1)

                if not args.not_data_init:
                    # Have the model and best model dict consistent with each other
                    # self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'      # Mode for testing if the base fc is replaced by the average feature vector
                    tsl, tsa = (test(self.model, testloader, 0, args, session, base_sess = True))[:2]
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()

                # Fine tuning on the novel class support set
                self.model.module.update_fc(trainloader, testloader, np.unique(train_set.targets), session)

                # Extending the support set of the novel classes with K exemplars each from the base session. And finetuning.
                joint_set, jointloader = get_jointloader(args, session)
                self.model.module.update_fc_joint(jointloader, testloader, np.unique(joint_set.targets), session)

                # tsl, tsa, tsaNovel, tsaBase, tsHm, tsAm, tcm, tcmNovel, tcmSummary, tsaPrevNovel = test(self.model, testloader, 0, args, session)        # Perform the validation inside the classifier updates and save the best mode/
                tsl, tsa, tsaNovel, tsaBase, tsHm, tsAm = test(self.model, testloader, 0, args, session)        # Perform the validation inside the classifier updates and save the best mode

                # Tensorboard logging
                self.writer.add_scalar('Accuracy (Top 1)', tsa, session)
                self.writer.add_scalars(f'Sessional Accuracy Split/', { # Novel and training accuracies uptil this particular session
                    "Novel Accuracy (Joint)":tsaNovel,   # Joint space accuracy for the current novel classes
                    "Base Accuracy (Joint)": tsaBase     # Joint space accuracy for the base classes
                }, session)
                self.writer.add_scalar('Harmonic Mean', tsHm, session)
                self.writer.add_scalar('Arithematic Mean', tsAm, session)
                # self.writer.add_figure("cm_joint", tcm["cm"], session)
                # self.writer.add_figure("cm_novel", tcmNovel["cm"], session)
                
                # Show the accuracy for the novel session in the previous session
                # if session > 1: # Note: This accuracy can be directly compared to the Sessional Accuracy Split/Novel Accuracy (Joint) from the previous session
                #     self.writer.add_scalars("Previous Session Accuracy Comparison", {
                #         "Before This Session": self.trlog['max_novel_acc'][session - 1] / 100,
                #         "After This Session": tsaPrevNovel
                #     }, session)
                    # self.writer.add_scalar("Previous Novel Classes Accuracy (Joint)", tsaPrevNovel, session)
                    
                # Get model weights and show in a tensorboard histogram
                for i in range(self.model.module.fc.weight.shape[0]):
                    self.writer.add_histogram(f"Classifier Vis {session}", self.model.module.fc.weight[i], i)

                # Save Accuracies and Means
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['max_novel_acc'][session] = float('%.3f' % (tsaNovel * 100))
                self.trlog['max_base_acc'][session] = float('%.3f' % (tsaBase * 100))
                self.trlog["max_hm"][session] = float('%.3f' % (hm(tsaBase, tsaNovel) * 100))
                self.trlog["max_am"][session] = float('%.3f' % (am(tsaBase, tsaNovel) * 100))


                # Save session-wise confusion matrix
                # tcm["cm"].savefig(os.path.join(self.cm_path, f"cm_{session}.png")) 
                # tcmNovel["cm"].savefig(os.path.join(self.cm_path, f"cmNovel_{session}.png"))

                # Save the best model
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)

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

        try:
            wandb.log({"avg_hm": average_harmonic_mean})
        except Exception as E:
            print(E)

        print("acc: ", self.trlog['max_acc'])
        print("hm: ", self.trlog['max_hm'])
        print("am: ", self.trlog['max_am'])
        print("base: ", self.trlog['max_base_acc'])
        print("novel: ", self.trlog['max_novel_acc'])
        print("avg_hm: ", average_harmonic_mean)

        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        self.writer.close()

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def sessional_test(self):
        args = self.args

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            if session == 0:  # load base class train img label
                pass
            else:
                print("training session: [%d]" % session)
                self.model.module.mode = self.args.new_mode
                self.model.eval()

                # Load sessions model_dict
                session_fp = os.path.join(self.args.experiment_path, f"session{session}_max_acc.pth")
                self.model.load_state_dict(torch.load(session_fp)['params'])

                tsl, tsa, tsaNovel, tsaBase, tsHm, tsAm, tcm, tcmNovel, tcmSummary = test(self.model, testloader, 0, args, session, max_inference = args.max_inference)        # Perform the validation inside the classifier updates and save the best mode                
                
                # Make directories
                if not os.path.exists(os.path.join(self.args.experiment_path, "confusion_matrix")):
                    os.mkdir(os.path.join(self.args.experiment_path, "confusion_matrix"))
                if not os.path.exists(os.path.join(self.args.experiment_path, "dataframes")):
                    os.mkdir(os.path.join(self.args.experiment_path, "dataframes"))

                # Save tcm and tcmNove into the files
                tcm["cm"].savefig(os.path.join(self.args.experiment_path, "confusion_matrix", f"cm_{session}.png")) 
                tcmNovel["cm"].savefig(os.path.join(self.args.experiment_path, "confusion_matrix", f"cmNovel_{session}.png"))
                tcmSummary["cm"].savefig(os.path.join(self.args.experiment_path, "confusion_matrix", f"cmSummary_{session}.png"))

                tcm["df"].to_csv(os.path.join(self.args.experiment_path, "dataframes", f"df_{session}.csv"))
                tcmNovel["df"].to_csv(os.path.join(self.args.experiment_path, "dataframes", f"dfNovel_{session}.csv"))

                out_string = 'Sess: {}, loss {:.3f}, testAcc {:.3f}, tsaNovel {:.3f}, tsaBase {:.3f}'\
                        .format(
                            session, 
                            tsl,
                            float('%.3f' % (tsa * 100.0)),
                            float('%.3f' % (tsaNovel * 100.0)),
                            float('%.3f' % (tsaBase * 100.0))
                            )

                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['max_novel_acc'][session] = float('%.3f' % (tsaNovel * 100))
                self.trlog['max_base_acc'][session] = float('%.3f' % (tsaBase * 100))

                print(out_string)

        print("Accuracies: ", self.trlog['max_acc'])
        print("Novel Accuracies: ", self.trlog['max_novel_acc'])
        print("Base Accuracies: ", self.trlog['max_base_acc'])


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
