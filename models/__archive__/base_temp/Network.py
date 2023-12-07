import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12_encoder import *
# import models.resnet18_lcwof as lcwof_net
from .helper import *

from tqdm import tqdm
from pdb import set_trace as bp

from utils import *

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
            # Total trainable parameters: 11207232 
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))s

        # if mode != "encode":    # Skip adding a fully connected layer if the mode for the base network class if encode
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)       # Note the entire number of classes are already added

        # Temeprature scaling variable. Set individually for each classifier.
        # self.temp = torch.ones(self.args.num_classes).cuda()
        self.base_temp = nn.Parameter(torch.ones(1)).cuda()
        self.novel_temp = nn.Parameter(torch.ones(1)).cuda()

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)) # Cosine classifier
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            # x = torch.div(self.fc(x), self.temp)
            logit = self.fc(x)
            logit[:, :self.args.base_class] = logit[:, :self.args.base_class] / self.base_temp
            logit[:, self.args.base_class:] = logit[:, self.args.base_class:] / self.novel_temp
            x = logit

        return x

    def encode(self, x):
        x = self.encoder(x)

        # TODO: Check if this is somehow effecting the output
        x = F.adaptive_avg_pool2d(x, 1)
        # x = self.avgpool(x)
        
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')
    
    def freeze_backbone(self):
        params_to_train = ['fc.weight']
        for name, param in self.named_parameters():
            # if name in params_to_train: 
            param.requires_grad = True if name in params_to_train else False

    def T_scaling(self, logits_list, labels_list, base_sess = True, novel_ix_start = 0):
        print("temperature scaling parameter being trained")
        # temperature = nn.Parameter(torch.ones(self.args.base_class))
        logits_list = torch.cat(logits_list).cuda()
        labels_list = torch.cat(labels_list).cuda()

        # temperature = nn.Parameter(torch.randn(self.args.num_classes, requires_grad=True, device="cuda"))
        # base_temp = nn.Parameter(torch.randn(1, requires_grad=True, device="cuda"))
        # novel_temp = nn.Parameter(torch.randn(1, requires_grad=True, device="cuda"))

        base_temp = nn.Parameter(torch.ones(1).cuda())
        novel_temp = nn.Parameter(torch.ones(1).cuda())

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
        optimizer = torch.optim.LBFGS([base_temp, novel_temp], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

        def _eval():
            temp = torch.ones(logits_list.shape[1]).cuda()
            temp[:self.args.base_class] = base_temp 
            temp[self.args.base_class:] = novel_temp

            loss = criterion(torch.div(logits_list, temp), labels_list)  # Without an applied softmax
            
            # # loss = 1/harmonicmean of logit score for novel and base classes 
            # m = nn.Softmax(dim = 1)

            # novel_mask = labels_list >= self.args.base_class       # The novel mask masks all classes which are from the novel sessions. Not necessarily the ones from the current session
            # base_mask = labels_list < self.args.base_class

            # # If no base classes then just optimise base_temp
            # if (novel_mask.type(torch.cuda.FloatTensor).sum() == 0).item():

            #     loss = criterion(torch.div(logits_list, base_temp), labels_list)
            #     return loss

            # # Applying individual temperatures to the individual classifiers
            # logits = torch.clone(logits_list)

            # # logits = torch.div(logits_list, temperature[:logits_list.shape[1]])
            # logits[:, :self.args.base_class] = logits[:, :self.args.base_class] / base_temp
            # logits[:, self.args.base_class:] = logits[:, self.args.base_class:] / novel_temp

            # pred = torch.argmax(m(logits), dim=1)[novel_mask]
            # label_ = labels_list[novel_mask]
            # novel_acc = (pred == label_).type(torch.cuda.FloatTensor)
            # novel_acc = novel_acc.mean()

            # pred = torch.argmax(m(logits), dim=1)[base_mask] 
            # label_ = labels_list[base_mask]
            # base_acc = (pred == label_).type(torch.cuda.FloatTensor)
            # base_acc = base_acc.mean()

            # loss = 1/am(base_acc, novel_acc)
            loss.backward()
            return loss

        # if not base_sess:
        #     # Seperate novel and base logits. And get two temperatures
        #     novel_mask = torch.argwhere(labels_list >= self.args.base_class).squeeze(1)
        #     labels_list = labels_list[novel_mask] - self.args.base_class
        #     logits_list = logits_list[novel_mask, self.args.base_class:]

        optimizer.step(_eval)

        self.base_temp = base_temp
        self.novel_temp = novel_temp

        # Create a temperature variable with a value for each single class. Currently for the novel classes 
        # self.temp = temperature
        # if base_sess:
        #     self.temp[:] = temperature.detach()
        # else:
        #     self.temp[novel_ix_start:novel_ix_start + self.args.way] = temperature.detach()

        # if base_sess:
        #     self.temp[:self.args.base_class] = temperature.detach()
        # else:
        #     self.temp[self.args.base_class:] = temperature.detach()

    def update_fc(self, trainloader, testloader, class_list, session):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            # >>> Setting a random set of parameters as the final classification layer for the novel classifier
            new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"),requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)
            
            # # Temperature scaling before the new_fc is trained
            if self.args.t_scaling:
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                concat_fc = torch.cat([old_fc, new_fc], dim=0)
                self.test_fc(concat_fc, testloader, 0, session, t_scaling = True)

        if 'ft' in self.args.new_mode:  # further finetune
            # self.update_fc_ft(new_fc,data,label,session)
            # Pass here also the testloader and args and sessio
            self.update_fc_ft_novel(new_fc, trainloader, testloader, session)

        # if self.args.t_scaling:
        #     old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
        #     concat_fc = torch.cat([old_fc, new_fc], dim=0)
        #     self.test_fc(concat_fc, testloader, 0, session, t_scaling = True)

        return

    def update_fc_avg(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            # Elemntwise division with temperature
            # output = F.linear(x,fc)
            # return torch.div(output, self.temp[:output.shape[1]])

            logit = F.linear(x, fc)
            logit[:, :self.args.base_class] = logit[:, :self.args.base_class] / self.base_temp
            logit[:, self.args.base_class:] = logit[:, self.args.base_class:] / self.novel_temp
            return logit

        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_new))
            for epoch in tqdm_gen:
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)

                loss = F.cross_entropy(logits, label) # Technically this is base normalised cross entropy. Because denominator has base sum and targets are only for the novel classes

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)
        return optimizer

    def update_fc_ft_novel(self, new_fc, trainloader, testloader, session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = self.get_optimizer_new(optimized_parameters)

        best_loss = None
        best_acc = None
        best_hm = None

        best_fc = None
        label_offset = (session - 1) * self.args.way
        
        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_new))
            old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for data, label in trainloader:
                    concat_fc = torch.cat([old_fc, new_fc], dim=0)
                    data = data.cuda()
                    label = label.cuda()

                    encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                    # if self.base_temp is not None:
                    #     old_logits = torch.div(self.get_logits(data, old_fc), self.base_temp)
                    #     new_logits = self.get_logits(data, new_fc)
                    #     logits = torch.cat([old_logits, new_logits], axis = 0)
                    # else:
                    logits = self.get_logits(encoding, concat_fc)

                    # The loss now in the denominator accounts for all the novel classes + base classes
                    loss = F.cross_entropy(logits, label)
                    
                    ta.add(count_acc(logits, label))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                vl, va, vaNovel, vaBase, vhm, vam = self.test_fc(concat_fc, testloader, epoch, session)
                if self.args.validation_metric == "hm":
                    # Validation
                    if best_hm is None or vhm > best_hm:
                        best_hm = vhm
                        best_fc = new_fc.clone()
                    out_string = '(Novel) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, vaNovel {:.3f}, vaBase {:.3f}'\
                        .format(
                            session, 
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0)),
                            float("%.3f" % (va * 100.0)),
                            float("%.3f" % (best_hm * 100.0)),
                            float("%.3f" % (vaNovel * 100.0)),
                            float("%.3f" % (vaBase * 100.0)),)
                elif self.args.validation_metric == "loss":
                    if best_loss is None or vl < best_loss:
                        best_loss = vl
                        best_fc = new_fc.clone()
                    out_string = 'Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                        .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)
                elif self.args.validation_metric == "acc":
                    if best_acc is None or va > best_acc:
                        best_acc = va
                        best_fc = new_fc.clone()
                    out_string = 'Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                        .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)

                tqdm_gen.set_description(out_string)
        
        # Update the weights
        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(best_fc.data)


    def update_fc_joint(self,jointloader, testloader, class_list, session):
        if 'ft' in self.args.new_mode:  # further finetune
            # self.update_fc_ft(new_fc,data,label,session)
            self.update_fc_ft_joint(jointloader, testloader, class_list, session)

    def update_fc_ft_joint(self, jointloader, testloader, class_list, session):
        """
            Extract the parameters associated with the given class list.
            So only base classifier and the novel classifier
            Creating a new classifier for the current classes and the new novel classes
            Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
            than 60 to be offset by the number of novel classes we have seen
        """

        novel_ix_start = self.args.base_class + self.args.way * (session - 1)
        novel_ix_end = self.args.base_class + self.args.way * (session)

        # Create a base fc and novel fc and copy these into the old fc to be trained together
        # By creating separate classifier we ensure that the intermediate fcs donot effect the output for the joint dataset
        base_fc = nn.Parameter(torch.zeros(self.args.base_class, self.num_features, device="cuda"), requires_grad=True)
        new_fc = nn.Parameter(torch.zeros(self.args.way, self.num_features, device="cuda"), requires_grad=True)
        inter_fc = self.fc.weight[self.args.base_class:self.args.base_class + (self.args.way * (session-1)), :].detach()

        # Set the values for the base and new fc from the model fc
        base_fc.data.copy_(self.fc.weight.data[:self.args.base_class, :])
        new_fc.data.copy_(self.fc.weight.data[novel_ix_start:novel_ix_end, :])

        optimized_parameters = [base_fc, new_fc]
        optimizer = self.get_optimizer_new(optimized_parameters)

        best_loss = None
        best_hm = None
        best_acc = None

        best_fc = None
        label_offset = (session - 1) * self.args.way

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_new))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for data, label in jointloader:
                    concat_fc = torch.cat([base_fc, inter_fc, new_fc], dim=0)
                    data = data.cuda()
                    label = label.cuda()

                    encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                    logits = self.get_logits(encoding, concat_fc)
                    
                    # label = label - label_offset # Offset labels only for labels with value greater than base_class
                    # label[label >= self.args.base_class] -= label_offset

                    loss = F.cross_entropy(logits, label)
                    
                    ta.add(count_acc(logits, label))

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                    optimizer.step()

                    total_loss += loss.item()


                # Model Saving
                vl, va, vaNovel, vaBase, vhm, vam = self.test_fc(concat_fc, testloader, epoch, session)
                if self.args.validation_metric == "hm":
                    # Validation
                    if best_hm is None or vhm > best_hm:
                        best_hm = vhm
                        best_fc = concat_fc.clone()
                    out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}'\
                        .format(
                            session, 
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0)),
                            float("%.3f" % (va * 100.0)),
                            float("%.3f" % (best_hm * 100.0)),)
                elif self.args.validation_metric == "loss":
                    if best_loss is None or vl < best_loss:
                        best_loss = vl
                        best_fc = concat_fc.clone()
                    out_string = 'Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                        .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)
                elif self.args.validation_metric == "acc":
                    if best_acc is None or va > best_acc:
                        best_acc = va
                        best_fc = concat_fc.clone()
                    out_string = 'Session {}, current_loss {:.3f}, train_acc {:.3f}'\
                        .format(session, total_loss,float('%.3f' % (ta.item() * 100.0)),)

                tqdm_gen.set_description(out_string)
        
        # Update the weights
        # self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(best_fc.data)
        self.fc.weight.data[:self.args.base_class, :].copy_(best_fc.data[:self.args.base_class, :])
        self.fc.weight.data[novel_ix_start:novel_ix_end, :].copy_(best_fc.data[novel_ix_start:novel_ix_end, :])


    def test_fc(self, fc, testloader, epoch, session, t_scaling = False):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        vl = Averager()
        va = Averager()

        # >>> Addition
        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only

        # test_fc = fc.clone().detach()
        all_targets=[]
        all_probs=[]

        with torch.no_grad():
            # tqdm_gen = tqdm(testloader)
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]
                encoding = self.encode(data).detach()

                logits = self.get_logits(encoding, fc)
                logits = logits[:, :test_class]

                all_targets.append(test_label)
                all_probs.append(logits)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                # >>> Addition
                novelAcc, baseAcc = count_acc_(logits, test_label, test_class, self.args)

                vaNovel.add(novelAcc)
                vaBase.add(baseAcc)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

            # >>> Addition 
            vaNovel = vaNovel.item()
            vaBase = vaBase.item()

        if t_scaling:
            self.T_scaling(all_probs, all_targets, base_sess=False, novel_ix_start=self.args.base_class + (session - 1) * self.args.way)

        vhm = hm(vaNovel, vaBase)
        vam = am(vaNovel, vaBase)

        return vl, va, vaNovel, vaBase, vhm, vam