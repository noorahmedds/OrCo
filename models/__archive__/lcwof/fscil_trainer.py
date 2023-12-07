from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
# from .Network import MYNET
import torch.nn.functional as F
from tqdm import tqdm

from models.base.Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)
        self.args = args
        self.set_save_path()                        # Setting the save path for the checkpoints for each run. 
        self.args = set_up_datasets(self.args)      # base_class, num_classes, way, shot and sessions defined. That are specific for each dataset. For example for miniimage net this is 60b, 100c, 5W, 5S, 9s
        self.set_up_model()                         # Setting up the model for base training. Resent is used. The depth depends on the dataset that is currently being trained on.

        self.trlog['max_novel_acc'] = [0.0] * args.sessions
        self.trlog['max_base_acc'] = [0.0] * args.sessions
        self.trlog['max_hm'] = [0.0] * args.sessions

        pass

    def set_up_model(self):
        # For lcwof we need just a simple fc layer at the end which is in dimension equal to the number of base classes (i.e. self.args.base_class)
        self.model = MYNET(self.args, mode=self.args.base_mode)                              # Using the base mode as the cos_ft (so fine tuning with cos application on the final layer)
        print(MYNET)

        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))        # Distribute model over defined gpus
        self.model = self.model.cuda()
        count_parameters(self.model)

        # In case we want to load from a pretrained model then we do that here. This is maybe important when you just want to test the incremental sessions and no retrain the base everytime
        if self.args.model_dir != None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']            # Loading the pretrained model that was trained on the base classes for 100 epochs.

        else:
            print('*********No Init model given. The model will be trained from scratch**********')
            # raise ValueError('You must initialize a pre-trained model')
            self.best_model_dict = None

            pass

    def update_param(self, model, pretrained_dict):
        if pretrained_dict:
            model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)

            # Note: The load state_dict loads the updated model dictionary while maintaining the require grad members
            model.load_state_dict(model_dict)

            if self.args.freeze_backbone:
                model.module.freeze_backbone()

        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = self.get_base_dataloader()
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
        return trainset, trainloader, testloader

    def get_base_dataloader(self):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                            index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                            index=class_index, base_sess=True)

        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                        index=class_index, base_sess=True)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)

        if self.args.dataset == 'mini_imagenet':
            # TODO: Follow lcwof with its transforms: https://github.com/Annusha/LCwoF/blob/f83dad1544b342323e33ea51f17bc03650e1e694/mini_imgnet/dataloader_classification.py#L44
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                index=class_index, base_sess=True)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_base, shuffle=True,
                                                num_workers=8, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        # Extracting data loader for the novel sessions
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=8, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=8, pin_memory=True)

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=True, #False
                                                 num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def get_utils_base(self, model):

        # optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
        #                              {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
        #                             momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        # optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': self.args.lr_base}],
        #                     momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)


        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        if self.args.criterion_base == "ce":
            criterion = nn.CrossEntropyLoss()

        return optimizer, scheduler, criterion

    def get_utils_novel(self, model):
        optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': self.args.lr_novel}],
                            momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)


        # criterion = CriterionAggregator()                       # The aggregator object accumulates multiple criterions iunternally if required and the forward function aggregates the contributing losses
        # if self.args.criterion_novel == "bnce":
        #     criterion.add_criterion(BNCrossEntropyLoss(self.args))
        # elif self.args.criterion_novel == "ce":
        #     criterion.add_criterion(nn.CrossEntropyLoss())

        # >>> More criterions get added here

        criterion = BNCrossEntropyLoss(self.args)
        if self.args.criterion_novel == "bnce":
            criterion = BNCrossEntropyLoss(self.args)
        elif self.args.criterion_novel == "ce":
            criterion = nn.CrossEntropyLoss()

        return optimizer, scheduler, criterion
        
    def train_base(self):
        """
            Params: 
                theta0: Pretrained model or a randomly initialised model that is used as the starting point for the base training phase.
        """
        args = self.args
        model = self.model
        result_list = []

        session = 0                                                         # Note: Session 0 points to the fact that this is the base training session and the model is going to be trained on the base classes defined by our index_list

        model = self.update_param(model, self.best_model_dict)              # Set pretrained model that is provided that is now being fine tuned
        optimizer, scheduler, criterion = self.get_utils_base(model)        # SGD Optimizer and a step scheduler from the torch lib
        train_set, trainloader, testloader = self.get_dataloader(session)   # Get data loader for our train step for the 0th session/base training step
        
        print('new classes for this session:\n', np.unique(train_set.targets))

        for epoch in range(args.epochs_base):
            start_time = time.time()

            tl = Averager()             # Loss Accumulator
            ta = Averager()             # Accuracy Accumulator

            # >>> Standard classificiation pretraining
            tqdm_gen = tqdm(trainloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, train_label = [_.cuda() for _ in batch]
                # TODO: Fix this
                preds = model(data)["base_logits"]                         # Predictions from the model only for the base logits
                # preds = model(data)                                      # Predictions from the model only for the base logits
                
                loss = criterion(preds, train_label)                       # loss calculation
                acc = count_acc(preds, train_label)                        # calculating accuracy

                lrc = scheduler.get_last_lr()[0]

                tqdm_gen.set_description('Phase (1), epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(epoch, lrc, loss.item(), acc))

                tl.add(loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tl = tl.item()
            ta = ta.item()

            # >>> Validation
            tsl, tsa = self.test_base(model, testloader, criterion)

            # Save better model
            if (tsa * 100) >= self.trlog['max_acc'][session]:
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['max_acc_epoch'] = epoch
                save_model_dir = os.path.join(args.save_path, 'base' + '_max_acc.pth')
                torch.save(dict(params=model.state_dict()), save_model_dir)
                torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                self.best_model_dict = deepcopy(model.state_dict())
                print('>>>A better model is found, test_acc:%.5f' % tsa)
                print('Saving model to :%s' % save_model_dir)
            
            # Per epoch metrics
            self.trlog['train_loss'].append(tl)
            self.trlog['train_acc'].append(ta)
            self.trlog['test_loss'].append(tsl)
            self.trlog['test_acc'].append(tsa)
            lrc = scheduler.get_last_lr()[0]
            result_list.append(
                'epoch:%03d, lr:%.4f, training_loss:%.5f, training_acc:%.5f, test_loss:%.5f, test_acc:%.5f' % (
                    epoch, lrc, tl, ta, tsl, tsa))

            print('This epoch takes %d seconds -- ' % (time.time() - start_time),
                    'still need around %.2f mins to finish this session' % (
                            (time.time() - start_time) * (args.epochs_base - epoch) / 60))
            
            scheduler.step()

        return result_list

    def train_novel(self):
        args = self.args
        model = self.model

        # final_session_model = torch.load("./checkpoint/mini_imagenet/lcwof/exp10-15W-1S-15Q-50Epi-L15W-L1SEpo_500-Lr1_0.100000-Lrg_0.10000-Step_40-Gam_0.10-T_16.00/incremental_session_8_max_acc.pth")
        # final_session_model = torch.load("./checkpoint/mini_imagenet/lcwof/exp11-15W-1S-15Q-50Epi-L15W-L1SEpo_500-Lr1_0.100000-Lrg_0.10000-Step_40-Gam_0.10-T_16.00/incremental_session_8_max_acc.pth")
        # final_session_model["params"]['module.fc.weight'] == model.module.fc.weight

        result_list = []

        theta1 = deepcopy(self.best_model_dict)                               # Deep copy to the base trained models weights for knowledge preservation module


        for session in range(1, args.sessions):
            model = self.update_param(model, self.best_model_dict)            # Take best model from previous incremental session / base training session
            optimizer, scheduler, criterion = self.get_utils_novel(model)

            # model.module.freeze_session_fc(session = session-1)           
            # model.module.freeze_session_fc_lt(session = session)            # Disallow the classification linear layers from the previous training session from being updated
            # model.module.add_fc(args.way)                                   # Add a new fc layer to the model for the new novel classes

            _, trainloader, testloader = self.get_dataloader(session)         # Load the training data relevent for this session

            for epoch in range(args.epochs_novel):
                tl = Averager()
                ta = Averager()
                model = model.train()

                tqdm_gen = tqdm(trainloader)
                for i, batch in enumerate(tqdm_gen, 1):
                    data, train_label = [_.cuda() for _ in batch]

                    preds = model(data)                                                                 # Predictions from the model

                    loss = criterion(preds, train_label, session)

                    # if not args.skip_kp:                                                              # Knowledge preservation weight constraint
                    #     loss += KP_weight_constraint_loss(theta1, model.state_dict(), args.lymbda_kp)

                    acc = count_acc(preds, train_label)                                                 # Calculating Accuracy, #TODO: Fix this as well

                    lrc = scheduler.get_last_lr()[0]

                    tqdm_gen.set_description(
                        'Phase (2), session {}, epoch {}, lrc={:.4f}, Loss={:.4f}, Top1 Accuracy={:.4f}'.format(session, epoch, lrc, loss.item(), acc))

                    tl.add(loss.item())
                    ta.add(acc)

                    optimizer.zero_grad()
                    loss.backward()

                    # Zeroing out gradients for base_classes because their weights contribute to the output of the final loss value 
                    # TODO. Kind of a hack. Fix this
                    # import pdb; pdb.set_trace()
                    model.module.fc.weight.grad[:args.base_class, :] = torch.zeros_like(model.module.fc.weight.grad[:args.base_class, :])

                    optimizer.step()

                tsl, tsa, tsaNovel, tsaBase = self.test_novel(model, testloader, args, session, criterion, theta1)
                thm = hm(tsaNovel, tsaBase)
                tam = am(tsaNovel, tsaBase)
            
                # Save best incremental model. That is the model with the highest harmonic mean
                # if (tsa * 100) >= self.trlog['max_acc'][session]: # Deprecated
                if (thm * 100) >= self.trlog['max_hm'][session]:    
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    self.trlog['max_novel_acc'][session] = float('%.3f' % (tsaNovel * 100))
                    self.trlog['max_base_acc'][session] = float('%.3f' % (tsaBase * 100))
                    self.trlog['max_hm'][session] = float('%.3f' % (thm * 100))

                    self.trlog['max_acc_epoch'] = epoch

                    # torch.save(dict(params=model.state_dict()), save_model_dir))
                    self.best_model_dict = deepcopy(model.state_dict())

                    print(">>> Found better model: best epoch {}, Harmonic Mean={:.3f}, Top1 Accuracy {:.3f}".format(self.trlog['max_acc_epoch'], thm*100, tsa*100)) #  am={:.3f}, hm={:.3f}

                print('>>> Testing: Top1 acc={:.3f}, Novel Accuracy={:.3f}, Base Accuracy={:.3f}, Harmonic Mean={:.3f}, Arithematic Mean={:.3f}'.format(
                    (tsa * 100), (tsaNovel*100), (tsaBase*100), (thm*100), (tam*100)))



            save_model_dir = os.path.join(args.save_path, 'incremental_session_' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.best_model_dict), save_model_dir)
            print('Saving model to :%s' % save_model_dir)

            best_am = am(self.trlog['max_novel_acc'][session], self.trlog['max_base_acc'][session])
            result_list.append('Session {}, Novel Acc (Joint)={:.3f}, Base Acc (Joint)={:.3f}, Harmonic Mean={:.3f}, Top1 Accuracy {:.3f}, Arithematic Mean={:.3f}\n'.format(
                                session, self.trlog['max_novel_acc'][session], self.trlog['max_base_acc'][session], self.trlog['max_hm'][session], self.trlog['max_acc'][session], best_am))

        return result_list

    def train_joint(self, theta2):
        # Take one random exemplar from each base training class and join with the novel training data.
        # For this we need to create a new data loader section
        # Where all sessions data is joined and a random training class exemplar is chosen.
        # Note the new target may now have classes belonging to all of the num_classes we have
        # Also note that our prediction will now be a concatenated single vector

        return theta2

    def test_base(self, model, testloader, criterion):
        model = model.eval()
        vl = Averager()
        va = Averager()

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits = model(data)
                logits = logits["output"]
                loss = criterion(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

        return vl, va

    def test_novel(self, model, testloader, args, session, criterion, theta1):
        # label_offset = args.base_class + ((session - 1) * args.way) # Label offset not required

        vl = Averager()
        va = Averager()
        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only        
        novel_class = args.base_class + (session - 1) * args.way

        model = model.eval()

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                logits = model(data)

                loss = criterion(logits, test_label, session)

                # if not args.not_kp:                                                     # Knowledge preservation weight constraint
                #     loss += KP_weight_constraint_loss(theta1, model.state_dict(), args.lymbda_kp)

                acc = count_acc(logits, test_label)
                novelAcc, baseAcc = count_acc_(logits, test_label, novel_class, args)
                vl.add(loss.item())
                va.add(acc)
                vaNovel.add(novelAcc)
                vaBase.add(baseAcc)

            vl = vl.item()
            va = va.item()
            vaNovel = vaNovel.item()
            vaBase = vaBase.item()

        return vl, va, vaNovel, vaBase

    def set_save_path(self):
        # self.args.save_path = "<self.args.dataset>/<self.args.project>/<wW-sS-qQ-epiEpi-LlW-LsS>"
        # for example: self.args.save_path = "mini_imagenet/cec/15W-1S-15Q-50Epi-L15W-L1S"
        # Setting the save path for all artefacts
        # TODO: Update to use Path lib for better readability
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path += "%s-" % (self.args.save_path_prefix)
        self.args.save_path = self.args.save_path + '%dW-%dS-%dQ-%dEpi-L%dW-L%dS' % (
            self.args.episode_way, self.args.episode_shot, self.args.episode_query, self.args.train_episode,
            self.args.low_way, self.args.low_shot)

        # Append the epochs, lr, lr for GraphAN, number of steps, gamma. temperature to the save path
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Lrg_%.5f-MS_%s-Gam_%.2f-T_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.lrg, mile_stone, self.args.gamma,
                self.args.temperature)
        elif self.args.schedule == 'Step': # <<< Default
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Lrg_%.5f-Step_%d-Gam_%.2f-T_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.lrg, self.args.step, self.args.gamma,
                self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path) # Make directories if not already there
        return None

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        if "1" in self.args.active_phase and self.best_model_dict is None:             # If a pretraining model is given the base session is skipped entirely
            # Phase 1 / Base Class Training:
            base_results = self.train_base()
            result_list.extend(base_results)

        if "2" in self.args.active_phase:
            # Phase 2 / Novel Class Training:
            novel_results = self.train_novel()
            result_list.extend(novel_results)


        # Phase 3 / Joint Training:
        # if "3" in self.args.active_phase:
        #     self.train_joint(self.model)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        