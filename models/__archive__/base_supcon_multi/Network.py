import sys
import os
sys.path.append(os.path.dirname(__file__))

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12_encoder import *
from models.alice_model import resnet_CIFAR
# import models.resnet18_lcwof as lcwof_net

from helper import *
from mixup import *

from tqdm import tqdm
from pdb import set_trace as bp

from utils import *
from scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR

from copy import deepcopy

from torch.autograd import Variable

from geom_median.torch import compute_geometric_median

import gaussian_utils

class SRHead(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        
        self.args = args
        self.num_features = num_features
        # self.novel_classifiers = [] # Here we store all the novel clasifiers

        # Classification head   
        if self.args.classifier_last_layer == "projection":
            self.base_fc = nn.Sequential(
                nn.Linear(self.num_features, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, self.args.base_class, bias=False)       # Note the entire number of classes are already added 
            )
        elif self.args.classifier_last_layer == "linear": 
            # if args.mixup_sup_con:
            #     extended_num_classes = int(self.args.base_class + (self.args.base_class * (self.args.base_class - 1))/2)
            #     self.base_fc = nn.Linear(self.num_features, extended_num_classes, bias=False)       # Note the entire number of classes are already added
            # else:   # default
            self.base_fc = nn.Linear(self.num_features, self.args.base_class, bias=False)       # Note the entire number of classes are already added

        self.classifiers = nn.Sequential(self.base_fc)
    
    def novel_requires_grad(self, value = True):
        for i, cls in enumerate(self.classifiers.children()):
            if isinstance(cls, nn.Sequential):
                cls[-1].weight.requires_grad = value
            elif isinstance(cls, nn.Sequential):
                cls.weight.requires_grad = value

    def forward(self, x):
        # Run output through all heads and output the final thing
        pass
    
    def append_novel_classifier(self, new_head):
        self.classifiers.append(new_head.cuda())
    
    def get_logits(self, encoding, session = 0):
        # Get logits corresponding to an encoding
        # Mask the gradients from the logits produced by session classifiers below session variable
        # If session == 0 then no classifier logits are masked
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            if 'dot' in self.args.new_mode:
                out = cls(encoding)
            elif 'cos' in self.args.new_mode:
                if isinstance(cls, nn.Sequential):
                    out = encoding
                    # Try the other way around. i.e. encoding normalised and then passed through to the classifier
                    for j, child in enumerate(cls.children()):
                        if j > 1: continue
                        out = child(out)
                    out = F.linear(F.normalize(out, p=2, dim=-1), F.normalize(cls[-1].weight, p=2, dim=-1)) 
                else:
                    out = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            
            out = out.detach() if i < session else out  # Only use during novel sessions. For joint session the session value is expected to be 0
            output.append(out)
        
        # Cat the outputs
        output = torch.cat(output, axis = 1)
        return output
        
class MYNET(nn.Module):

    def __init__(self, args, mode=None, writer=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            # self.encoder = resnet20()
            # self.encoder_outdim = 64

            self.encoder = resnet_CIFAR.ResNet18()
            self.encoder.fc = nn.Identity()
            self.encoder_outdim = 512

        if self.args.dataset in ['mini_imagenet']:
            # self.encoder = resnet18(False, args)  # pretrained=False
            # self.encoder_outdim = 512
            # Total trainable parameters: 11207232 

            # ALICE
            self.encoder = resnet_CIFAR.ResNet18()  # out_dim = 128
            self.encoder.fc = nn.Identity()
            self.encoder_outdim = 512

        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.encoder_outdim = 512

        self.writer = writer

        # Classification head   
        # if self.args.classifier_last_layer == "projection":
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.num_features, 2048),
        #         nn.ReLU(),
        #         nn.Linear(2048, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, self.args.num_classes, bias=False)       # Note the entire number of classes are already added 
        #     )
        # elif self.args.classifier_last_layer == "linear":
        #     self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)       # Note the entire number of classes are already added

        if self.args.use_sup_con_head:
            # Credit: https://github.com/HobbitLong/SupContrast/blob/331aab5921c1def3395918c05b214320bef54815/networks/resnet_big.py#L174
            self.sup_con_head = nn.Sequential(
                nn.Linear(self.encoder_outdim, self.encoder_outdim),
                nn.ReLU(inplace=True),  # Relu works here because the linear layer is being trained from scratch. Our reserved prototypes have negative values in them
                nn.Linear(self.encoder_outdim, self.args.sup_con_feat_dim)
            )
        else:
            self.sup_con_head = nn.Linear(self.encoder_outdim, self.args.sup_con_feat_dim)
            # self.sup_con_head = nn.Linear(self.num_features, self.num_features) # num_features as the number of output as this will then be used as the init for base

        if self.args.ce_pretrain:
            self.ce_head = nn.Linear(self.encoder_outdim, self.args.base_class)

        self.num_reserve = 100
        self.reserved_prototypes = None

        if self.args.incremental_on_supcon_head:
            self.num_features = self.args.sup_con_feat_dim
        else:
            self.num_features = self.encoder_outdim

        self.fc = SRHead(self.args, self.num_features)

        # Class aug
        # extended_num_classes = int(self.args.base_class + (self.args.base_class * (self.args.base_class - 1))/2)
        # self.fc = nn.Linear(self.num_features, extended_num_classes, bias=False)       # Note the entire number of classes are already added

        self.toggle_sup_con = False

    def forward_sup_con_mixup(self, x, **kwargs):
        out = self.encoder(x, **kwargs)
        x, y_a, y_b, lam = out
        x = F.adaptive_avg_pool2d(x, 1)
        encoding = x.squeeze(-1).squeeze(-1)
        encoding = F.normalize(encoding, dim=1)
        x = F.normalize(self.sup_con_head(encoding), dim=1)
        return x, y_a, y_b, lam, encoding

    def forward_sup_con(self, x, **kwargs):
        if self.args.skip_encode_norm:
            x = self.encode(x)
        else:  
            x = F.normalize(self.encode(x), dim = 1)

        if self.args.skip_sup_con_head:
            return x
        
        x = F.normalize(self.sup_con_head(x), dim=1)

        # TODO: If the reservation method is positive hemisphere then we force the second axis to always be posititve
        # This should ensure that the second element of all vectors during the base session will be positive. 
        # And given that this function is used only during the base session should ensure that this does not happen during inc
        # if self.args.reserve_method == "hemisphere":
        #     x[:,1] = torch.abs(x[:, 1])
        # This shuld be done in the loss function
        
        return x
        
    def forward_ce(self, x):
        encoding = self.encode(x)
        x = self.ce_head(encoding)
        return x

    def forward_metric(self, x):
        encoding = self.encode(x)

        if "freeze" in self.mode:
            encoding = encoding.detach()

        x = self.fc.get_logits(encoding, 0)

        return x, encoding

    def encode(self, x):
        x = self.encoder(x)

        if self.args.dataset not in ["mini_imagenet", "cifar100"]:  # Cub model requires average ppooling to happen here
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        if self.toggle_sup_con:
            x = self.sup_con_head(x)

        return x

    def forward_mixup(self, input, **kwargs):
        out = self.encoder(input, **kwargs)
        
        # Unpacking from mixup
        x, y_a, y_b, lam = out

        x = F.adaptive_avg_pool2d(x, 1)
        encoding = x.squeeze(-1).squeeze(-1)

        if 'cos' in self.mode:
            x = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)) # Cosine classifier
            x = self.args.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(encoding)

        return x, y_a, y_b, lam, encoding

    def forward(self, input, **kwargs):
        if self.mode == "sup_con":
            return self.forward_sup_con(input)
        elif self.mode == "cross_entropy":
            return self.forward_ce(input)
        elif self.mode == "sup_con_mixup":
            return self.forward_sup_con_mixup(input, **kwargs)
        
        if self.mode != 'encoder':
            input, encoding = self.forward_metric(input)
            return input, encoding
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

    def update_fc(self, trainloader, testloader, class_list, session):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            # >>> Setting a random set of parameters as the final classification layer for the novel classifier
            # new_head = nn.Parameter(torch.rand(len(class_list), self.num_features, device="cuda"),requires_grad=True)
            # nn.init.kaiming_uniform_(new_head, a=math.sqrt(5))
            new_head = nn.Linear(self.num_features, len(class_list), device="cuda") # Random initialisation
        else:
            if self.args.proto_method == "mean":
                new_head = self.update_fc_avg(data, label, class_list)
            elif self.args.proto_method == "gm":
                new_head = self.update_fc_gm(data, label, class_list)

            # create a new fc with a projection head behind it and the new fc itself without any gradients
            # new_head = self.update_fc_sr(session)

        self.fc.append_novel_classifier(new_head)

        if 'ft' in self.args.new_mode:  # further finetune
            # self.update_fc_ft(new_fc,data,label,session)
            # Pass here also the testloader and args and session
            if not self.args.skip_novel:
                return self.update_fc_ft_novel(trainloader, testloader, session)

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
            # self.fc.weight.data[class_index]=proto

            # TODO:
            # Maximise separation from the previous prototypes

        # Note the original protonet sums the latent vectors over the support set, and divides by the number of classes. Not by the number of data points
        if not self.args.skip_encode_norm: # test
            new_fc_tensor=F.normalize(torch.stack(new_fc,dim=0), dim=1)
        else:
            new_fc_tensor=torch.stack(new_fc,dim=0)

        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0]).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        # new_fc.weight.requires_grad = False
        return new_fc

    def update_fc_gm(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            # proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            # Calculate geometric median instead
            proto = compute_geometric_median(embedding.cpu()).median.cuda()
            # proto = compute_geometric_mean(embedding)

            new_fc.append(proto)
            # self.fc.weight.data[class_index]=proto

            # TODO:
            # Maximise separation from the previous prototypes

        # Note the original protonet sums the latent vectors over the support set, and divides by the number of classes. Not by the number of data points
        if not self.args.skip_encode_norm:
            new_fc_tensor = F.normalize(torch.stack(new_fc,dim=0), dim=1)
        else:
            new_fc_tensor = torch.stack(new_fc,dim=0)

        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0]).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        # new_fc.weight.requires_grad = False
        return new_fc

    def update_fc_sr(self,session):
        """
            Using some reserved pre generated prototypes and assigning some to the new fc final layer with a projection head behind it 
        """
        # Assign to the new_fc the chosen prototypes
        start_ix = self.args.way * (session - 1) # (1-1) * 5 = 0 5
        end_ix = self.args.way * session         # 1 * 5 = 5 10

        # final_layer_weights = self.reserved_prototypes[start_ix:end_ix]
        if start_ix == 0:
            final_layer_weights = self.reserved_prototypes[-end_ix:]
        else:
            final_layer_weights = self.reserved_prototypes[-end_ix:-start_ix]
    
        # Old Definition Add more lienar layers here to see whether the model is able to learn a proper projection
        if self.args.novel_high_layers:
            new_head = nn.Sequential(
                # Extra layer
                nn.Linear(self.num_features, self.num_features*2),
                nn.ReLU(inplace=True),  # Note that the relu will completely remove all negative components of the projection. 
                nn.Linear(self.num_features*2, self.num_features),
                nn.ELU(inplace=True),  # Note that the relu will completely remove all negative components of the projection. 
                nn.Linear(self.num_features, self.args.way, bias=False) # output
            )
        else:
            if self.args.skip_nonlinearity:
                new_head = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.Linear(self.num_features, self.args.way, bias=False) # output
            )
            else:
                new_head = nn.Sequential(
                    nn.Linear(self.num_features, self.num_features),
                    nn.ELU(inplace=True),  
                    # nn.ReLU(inplace=True),  # Note that the relu will completely remove all negative components of the projection. 
                    # nn.Tanh(),  # Note that the relu will completely remove all negative components of the projection. 
                    # nn.GELU(),
                    nn.Linear(self.num_features, self.args.way, bias=False) # output
                )       

        new_head[-1].weight.data.copy_(final_layer_weights)
        new_head[-1].weight.requires_grad = False
        
        return new_head

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
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

    def finetune_head(self, loader, epochs=None):
        # Only for testing purposes
        # fine tuning the head given some data and labels
        optimizer = self.get_optimizer_new(self.fc.parameters())

        with torch.enable_grad():

            loss_before = 0
            for i, batch in enumerate(loader):
                data, label = [_.cuda() for _ in batch]
                encoding = self.encode(data).detach()
                logits = self.fc.get_logits(encoding)
                loss_before += F.cross_entropy(logits, label)
            loss_before /= len(loader)
            print(f"Loss at the beginning of training: {loss_before.item()}")

            tqdm_gen = tqdm(range(self.args.epochs_new))
            for epoch in tqdm_gen:
                for i, batch in enumerate(loader):
                    data, label = [_.cuda() for _ in batch]
                    encoding = self.encode(data).detach()
                    logits = self.fc.get_logits(encoding)

                    loss = F.cross_entropy(logits, label) # Technically this is base normalised cross entropy. Because denominator has base sum and targets are only for the novel classes

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tqdm_gen.set_description(f"Loss: {loss.item():.3f}")

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9 if not self.args.nesterov_new else 0, weight_decay=self.args.decay_new, nesterov=self.args.nesterov_new)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)
        return optimizer

    def update_fc_ft_novel(self, trainloader, testloader, session):
        # new_head=new_head.detach().clone()
        # optimized_parameters = [{'params': new_head[:-1]}] # everything gets optimised except the final layer
        # optimized_parameters = [{'params': self.fc.parameters()}] # everything gets optimised except the final layer
        optimizer = self.get_optimizer_new(self.fc.parameters())

        criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_novel)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],
                                                             gamma=self.args.gamma)

        best_loss = None
        best_acc = None
        best_hm = None

        best_fc = None
        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        average_gap = Averager()
        average_gap_n = Averager()
        average_gap_b = Averager()


        # Before training accuracies
        val_freq = 5
        vl, va, vaNovel, vaBase, vhm, vam, vbin = self.test_fc_head(self.fc, testloader, 0, session)
        print(f"Top 1 Acc: {va*100:.3f}, N/j Acc: {vaNovel*100:.3f}, B/j Acc: {vaBase*100:.3f}, Binary Accuracy: {vbin*100:.3f}")

        self.eval() # Fixing batch norm

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_novel))
            # old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()

            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for data, label in trainloader:
                    # concat_fc = torch.cat([old_fc, new_fc], dim=0)
                    data = data.cuda()
                    label = label.cuda()
                    label[label >= self.args.base_class] -= label_offset    # Offsetting the label to match the fc ouptut

                    if self.args.mixup_novel:
                        # data will now contain the mixed up novel session data
                        data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                    encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                    # logits = self.get_logits(encoding, concat_fc)
                    logits = self.fc.get_logits(encoding, session) # In the joint session we just pass 0 for session because there we optimise everything       Top 1 Acc: 49.754, N/j Acc: 46.200, B/j Acc: 50.050, Binary Accuracy: 64.108
                    # logits = self.fc.get_logits(encoding, 0) # This should not work because we then forget too much                                           Top 1 Acc: 48.662, N/j Acc: 46.400, B/j Acc: 48.850, Binary Accuracy: 62.246

                    # Take the maximum logit score in the novel classifier and the maximum logit score in the previous sets of classes find the different in the logit outputs. 
                    # Compute this distance for each batch sample and print
                    for i in range(logits.shape[0]):
                        average_gap.add(logits[i][self.args.base_class + (self.args.way * (session-1)):].max() - logits[i][:self.args.base_class].max())

                    novel_logits = logits[:, self.args.base_class + (self.args.way * (session-1)):]
                    base_logits = logits[:, :self.args.base_class]
                    average_gap_n.add((novel_logits.max(axis=1)[0] - novel_logits.min(axis=1)[0]).mean())
                    average_gap_b.add((base_logits.max(axis=1)[0] - base_logits.min(axis=1)[0]).mean())


                    if self.args.mixup_novel:
                        # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        # The loss now in the denominator accounts for all the novel classes + base classes
                        # loss = F.cross_entropy(logits, label) # Note, no label smoothing here
                        loss = criterion(logits, label)
                    
                    ta.add(count_acc(logits, label))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()


                # Model Saving
                if self.args.validation_metric_novel == "none":
                    out_string = '(Novel) Session {}, current_loss {:.3f}, train_acc {:.3f}, Average Logit Gap {:.3f}, ALGb/n {:.3f}/{:.3f}'\
                        .format(session, 
                                total_loss, 
                                float('%.3f' % (ta.item() * 100.0)), 
                                float('%.3f' % (average_gap.item() * 100.0)),
                                float('%.3f' % (average_gap_b.item() * 100.0)),
                                float('%.3f' % (average_gap_n.item() * 100.0))
                                )
                    if epoch % val_freq == 0:
                        vl, va, vaNovel, vaBase, vhm, vam, vbin = self.test_fc_head(self.fc, testloader, epoch, session)
                        s = (f"Top 1 Acc: {va*100:.3f}, "
                            f"N/j Acc: {vaNovel*100:.3f}, "
                            f"B/j Acc: {vaBase*100:.3f}, "
                            f"Binary Accuracy: {vbin*100:.3f}")
                        print(s)
                        # TODO: Print binary classification score
                        # TODO: Measure false positives in the base classifier alone. 
                        
                tqdm_gen.set_description(out_string)

                # self.writer.add_scalars(f"(Novel) Session {session} Training Graph", {
                #     "nta": ta.item(),
                #     "nva": 0,
                # }, epoch)

                scheduler.step()


    def update_fc_joint(self,jointloader, testloader, class_list, session, data_previous=None, generated = None):
        if generated is None:
            return self.update_fc_ft_joint(jointloader, testloader, class_list, session, data_previous)
        else:
            if self.args.mixup_joint:
                return self.update_fc_ft_joint_gaussian_mixup(jointloader, testloader, class_list, session, generated)
            else:
                return self.update_fc_ft_joint_gaussian(jointloader, testloader, class_list, session, generated)

    def update_fc_ft_joint(self, jointloader, testloader, class_list, session, data_previous):
        """
            Extract the parameters associated with the given class list.
            So only base classifier and the novel classifier
            Creating a new classifier for the current classes and the new novel classes
            Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
            than 60 to be offset by the number of novel classes we have seen
        """

        # optimized_parameters = [base_fc, new_fc]
        if self.args.train_novel_in_joint:
            self.fc.novel_requires_grad(value=True)

        if self.args.fine_tune_backbone_joint:
            # optimizer = torch.optim.SGD([
            #     {'params': self.fc.parameters()},
            #     {'params': self.encoder.layer4.parameters(), 'lr': 5e-4}
            # ], lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)

            # Train everything
            optimizer = torch.optim.SGD([
                {'params': self.fc.parameters()},
                {'params': self.encoder.layer4.parameters(), 'lr': 5e-4}
            ], lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
            
            # create ewc object
            # Previous session data and the current model are passed
            ewc = EWC(self, data_previous, self.args.way)
            importance = 100   # high importance to the previous task

        else:
            optimizer = self.get_optimizer_new(self.fc.parameters())


        # TODO: Try different optimiser
        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
        else:
            warmup_epochs = 10
            min_lr = 0.0
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=10,
                    max_epochs=self.args.epochs_joint,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                    eta_min=min_lr
                )

        criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

        best_loss = None
        best_hm = None
        best_acc = None

        best_fc = None
        best_l4 = None
        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        hm_patience = self.args.hm_patience
        hm_patience_count = 0
        
        average_gap = Averager()
        average_gap_n = Averager()
        average_gap_b = Averager()
        
        novel_class_start = self.args.base_class + (self.args.way * (session-1))

        self.eval() # Fixing batch norm

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                ewc_loss = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for data, label in jointloader:
                    data = data.cuda()
                    label = label.cuda()
                    label[label >= self.args.base_class] -= label_offset

                    if self.args.mixup_joint:
                        data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                        # TODO: Mixup only within class    

                    if self.args.fine_tune_backbone_joint:
                        encoding = self.encode(data)
                    else:
                        encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                    # logits = self.get_logits(encoding, concat_fc)
                    logits = self.fc.get_logits(encoding)   

                    # Only for targets of novel classes
                    novel_targets = label > (self.args.base_class + (self.args.way * (session-1)))
                    if True in novel_targets:
                        novel_logits = logits[novel_targets, self.args.base_class + (self.args.way * (session-1)):]
                        base_logits = logits[novel_targets, :self.args.base_class]

                        average_gap.add((novel_logits.max(axis=1)[0] - base_logits.max(axis=1)[0]).mean())
                        average_gap_n.add((novel_logits.max(axis=1)[0] - novel_logits.min(axis=1)[0]).mean())
                        average_gap_b.add((base_logits.max(axis=1)[0] - base_logits.min(axis=1)[0]).mean())
                    
                    # label = label - label_offset # Offset labels only for labels with value greater than base_class
                    # label[label >= self.args.base_class] -= label_offset

                    if self.args.mixup_joint:
                        # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        # The loss now in the denominator accounts for all the novel classes + base classes
                        # TODO: toggle on
                        # loss = criterion(logits, label) # Note, no label smoothing here

                        # import pdb; pdb.set_trace()
                        # Calculate the loss for novel classes only
                        # This is to cater towards the bias towards base classes when training in the joint session.
                        # This way we train equally the base and novel classifiers free from any bias. 
                        # This is a must in our technique
                        # TODO: Average the losses for each incremental session
                        if self.args.joint_loss in ['ce_even', 'ce_inter']:
                            losses = []
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            if self.args.joint_loss == "ce_even":
                                base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                            elif self.args.joint_loss == "ce_inter":
                            # TODO: For new sessions weight everybody together
                                inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                                base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                                if inter_classes_idx.numel() != 0:
                                    inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                    losses.append(inter_loss)

                            if novel_classes_idx.numel() != 0:
                                # Loss computed using the novel classes
                                novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                                losses.append(novel_loss)

                            if base_classes_idx.numel() != 0:
                                base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                                losses.append(base_loss)

                            loss = 0
                            for l in losses: loss += l
                            loss /= len(losses)
                        elif self.args.joint_loss == "ce_weighted":
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                            w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                            w_replay = 1-w_current          # 0.5, 0.667, ...

                            loss = 0
                            if novel_classes_idx.numel() != 0:
                                novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                                loss += w_current * novel_loss

                            if base_classes_idx.numel() != 0:
                                base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                                loss += w_replay * base_loss
                        else:
                            # Original ce
                            loss = criterion(logits, label)
                    
                    if self.args.fine_tune_backbone_joint:
                        # add ewc loss to losses
                        el = importance * ewc.penalty(self)
                        ewc_loss.add(el)
                        loss += el

                    ta.add(count_acc(logits, label))

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                vl, va, vaNovel, vaBase, vhm, vam, vbin = self.test_fc_head(self.fc, testloader, epoch, session)
                if self.args.validation_metric == "hm":
                    # Validation
                    if best_hm is None or vhm > best_hm:
                        best_hm = vhm
                        # best_fc = concat_fc.clone()
                        # best_fc = concat_fc.detach().clone()
                        best_fc = deepcopy(self.fc.state_dict())
                        if self.args.fine_tune_backbone_joint:
                            # best_l4 = deepcopy(self.encoder.state_dict())
                            best_l4 = deepcopy(self.encoder.layer4.state_dict())
                        hm_patience_count = 0
                    else:
                        hm_patience_count += 1
                    out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, ewc_loss={:.3f}, (test) b/n={:.3f}/{:.3f}, ALG {:.1}, ALGb/n {:.1f}/{:.1f}'\
                        .format(
                            session, 
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0)),
                            float("%.3f" % (va * 100.0)),
                            float("%.3f" % (best_hm * 100.0)),
                            float("%.3f" % (ewc_loss.item())),
                            float("%.3f" % (vaBase * 100.0)),
                            float("%.3f" % (vaNovel * 100.0)),
                            float('%.3f' % (average_gap.item() * 100.0)),
                            float('%.3f' % (average_gap_b.item() * 100.0)),
                            float('%.3f' % (average_gap_n.item() * 100.0))
                            )

                tqdm_gen.set_description(out_string)

                # self.writer.add_scalars(f"(Joint) Session {session} Training Graph", {
                #     "jta": ta.item(),
                #     "jva": va,
                #     # "jvaNovel": vaNovel
                # }, epoch)

                # import pdb; pdb.set_trace()

                if hm_patience_count > hm_patience:
                    # faster joint session
                    break
                
                scheduler.step()
        
        # # Update the weights
        # if self.args.base_only:
        #     self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(best_fc.data[-self.args.way:, :])
        #     self.fc.weight.data[:self.args.base_class, :].copy_(best_fc.data[:self.args.base_class, :])
        #     # self.fc.weight.data[novel_ix_start:novel_ix_end, :].copy_(best_fc.data[novel_ix_start:novel_ix_end, :])
        # else:
        #     with torch.no_grad(): self.fc.weight[:self.args.base_class + self.args.way * session, :] = nn.Parameter(best_fc)
        self.fc.load_state_dict(best_fc, strict=True)
        if best_l4 is not None:
            # self.encoder.load_state_dict(best_l4)
            self.encoder.layer4.load_state_dict(best_l4)

    def update_fc_ft_joint_gaussian_loader(self, jointloader, testloader, class_list, session):
        """
            Extract the parameters associated with the given class list.
            So only base classifier and the novel classifier
            Creating a new classifier for the current classes and the new novel classes
            Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
            than 60 to be offset by the number of novel classes we have seen
        """
        if self.args.train_novel_in_joint:
            self.fc.novel_requires_grad(value=True)

        optimizer = self.get_optimizer_new(self.fc.parameters())

        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
        else:
            warmup_epochs = 10
            min_lr = 0.0
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=10,
                    max_epochs=self.args.epochs_joint,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                    eta_min=min_lr
                )

        criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

        best_loss = None
        best_hm = None
        best_acc = None

        best_fc = None
        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        hm_patience = self.args.hm_patience
        hm_patience_count = 0
        
        novel_class_start = self.args.base_class + (self.args.way * (session-1))
        test_class = self.args.base_class + session * self.args.way

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))

            # To emulate augmentation in generated data we need to mixup the generated data
            # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                taNovel = Averager()
                taBase = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for i, (data, gauss, label, ei_flag) in enumerate(jointloader):
                    encoding = torch.empty(0, gauss.shape[1]).cuda()

                    data = data[ei_flag == 1]   # Filter the image data
                    data = data.cuda()
                    label = label.cuda()

                    gauss = gauss[ei_flag == 0].cuda()

                    label[label >= self.args.base_class] -= label_offset

                    # Filter data
                    if data.numel() > 0:
                        encoding = self.encode(data).detach() #<<< The encoder is essentially frozen
                    
                    if gauss.numel() > 0:
                        encoding = torch.cat((gauss, encoding))

                    # Combine with the encodings in this batch
                    label = torch.cat((label[ei_flag == 0], label[ei_flag == 1]))

                    logits = self.fc.get_logits(encoding) 

                    if self.args.mixup_joint:
                        # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        if self.args.joint_loss in ['ce_even', 'ce_inter']:
                            losses = []
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            if self.args.joint_loss == "ce_even":
                                base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                            elif self.args.joint_loss == "ce_inter":
                                inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                                base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                                if inter_classes_idx.numel() != 0:
                                    inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                    losses.append(inter_loss)

                            if novel_classes_idx.numel() != 0:
                                novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                                losses.append(novel_loss)

                            if base_classes_idx.numel() != 0:
                                base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                                losses.append(base_loss)

                            loss = 0
                            for l in losses: loss += l
                            loss /= len(losses)
                        elif self.args.joint_loss == "ce_weighted":
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                            w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                            w_replay = 1-w_current          # 0.5, 0.667, ...

                            loss = 0
                            if novel_classes_idx.numel() != 0:
                                novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                                loss += w_current * novel_loss

                            if base_classes_idx.numel() != 0:
                                base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                                loss += w_replay * base_loss
                        else:
                            # Original ce
                            loss = criterion(logits, label) # Note, no label smoothing here
                    
                    ta.add(count_acc(logits, label))
                    n,b = count_acc_(logits, label, test_class, self.args)
                    taNovel.add(n)
                    taBase.add(b)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                vl, va, vaNovel, vaBase, vhm, vam, vbin = self.test_fc_head(self.fc, testloader, epoch, session)
                if self.args.validation_metric == "hm":
                    # Validation
                    if best_hm is None or vhm > best_hm:
                        best_hm = vhm
                        best_fc = deepcopy(self.fc.state_dict())
                        hm_patience_count = 0
                    else:
                        hm_patience_count += 1
                    lrc = scheduler.get_last_lr()[0]
                    out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (train) b/n:{:.3f}/{:.3f}, (test) b/n={:.3f}/{:.3f}, lrc={:.3f}'\
                        .format(
                            session, 
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0)),
                            float("%.3f" % (va * 100.0)),
                            float("%.3f" % (best_hm * 100.0)),
                            float("%.3f" % (taBase.item() * 100.0)),
                            float("%.3f" % (taNovel.item() * 100.0)),
                            float("%.3f" % (vaBase * 100.0)),
                            float("%.3f" % (vaNovel * 100.0)),
                            float(lrc))
                tqdm_gen.set_description(out_string)

                if hm_patience_count > hm_patience:
                    # faster joint session
                    break

                scheduler.step()
        
        self.fc.load_state_dict(best_fc, strict=True)   

    def update_fc_ft_joint_gaussian(self, jointloader, testloader, class_list, session, generated):
        """
            Extract the parameters associated with the given class list.
            So only base classifier and the novel classifier
            Creating a new classifier for the current classes and the new novel classes
            Note that this new classifier is a contiguous classifier so it needs to be trained with taregts greater 
            than 60 to be offset by the number of novel classes we have seen
        """

        gaus, aug_gaus = generated
        gaussian_data, gaussian_labels = gaus
        aug_gaus_data, aug_gaus_labels = aug_gaus

        # optimized_parameters = [base_fc, new_fc]
        if self.args.train_novel_in_joint:
            self.fc.novel_requires_grad(value=True)

        optimizer = self.get_optimizer_new(self.fc.parameters())

        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
        else:
            warmup_epochs = 10
            min_lr = 0.0
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=10,
                    max_epochs=self.args.epochs_joint,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                    eta_min=min_lr
                )

        criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

        best_loss = None
        best_hm = None
        best_acc = None

        best_fc = None
        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        hm_patience = self.args.hm_patience
        hm_patience_count = 0

        # shuffle generated data
        shuffler = np.arange(gaussian_data.shape[0])
        np.random.shuffle(shuffler)
        gaussian_data = gaussian_data[shuffler]
        gaussian_labels = gaussian_labels[shuffler]
        
        # split equally
        total_iter = jointloader.__len__()
        generated_data = np.array_split(gaussian_data, total_iter)
        generated_labels = np.array_split(gaussian_labels, total_iter)

        novel_class_start = self.args.base_class + (self.args.way * (session-1))
        test_class = self.args.base_class + session * self.args.way

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))

            # To emulate augmentation in generated data we need to mixup the generated data
            # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                taNovel = Averager()
                taBase = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for i, (data, label) in enumerate(jointloader):
                    data = data.cuda()
                    label = label.cuda()
                    label[label >= self.args.base_class] -= label_offset

                    if self.args.mixup_joint:
                        data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                    encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                    # Concatenate the encoding with the generated_data and the labels as well
                    # encoding = torch.cat((torch.tensor(generated_data[i]).cuda(), encoding)).to(torch.float)
                    # label = torch.cat((torch.tensor(generated_labels[i]).cuda(), label)).to(torch.long)

                    # Before concatenating generated label, each item gets augmented with a random sample from generated_data
                    augmented_data = gaussian_utils.augmentMultivariateData(generated_data[i], generated_labels[i], aug_gaus_data, aug_gaus_labels)
                    encoding = torch.cat((torch.tensor(augmented_data).cuda(), encoding)).to(torch.float)
                    label = torch.cat((torch.tensor(generated_labels[i]).cuda(), label)).to(torch.long)
                    logits = self.fc.get_logits(encoding)

                    if self.args.mixup_joint:
                        # loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        if self.args.joint_loss in ['ce_even', 'ce_inter']:
                            losses = []
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            if self.args.joint_loss == "ce_even":
                                base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                            elif self.args.joint_loss == "ce_inter":
                            # TODO: For new sessions weight everybody together
                                inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                                base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                                if inter_classes_idx.numel() != 0:
                                    inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                    losses.append(inter_loss)

                            if novel_classes_idx.numel() != 0:
                                # Loss computed using the novel classes
                                novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                                losses.append(novel_loss)

                            if base_classes_idx.numel() != 0:
                                base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                                losses.append(base_loss)

                            loss = 0
                            for l in losses: loss += l
                            loss /= len(losses)
                        elif self.args.joint_loss == "ce_weighted":
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                            w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                            w_replay = 1-w_current          # 0.5, 0.667, ...

                            loss = 0
                            if novel_classes_idx.numel() != 0:
                                novel_loss = criterion(logits[novel_classes_idx, :], label[novel_classes_idx])
                                loss += w_current * novel_loss

                            if base_classes_idx.numel() != 0:
                                base_loss = criterion(logits[base_classes_idx, :], label[base_classes_idx])
                                loss += w_replay * base_loss
                        else:
                            # Original ce
                            loss = criterion(logits, label) # Note, no label smoothing here
                    
                    ta.add(count_acc(logits, label))

                    n,b = count_acc_(logits, label, test_class, self.args)
                    taNovel.add(n)
                    taBase.add(b)

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                vl, va, vaNovel, vaBase, vhm, vam, vbin = self.test_fc_head(self.fc, testloader, epoch, session)
                if self.args.validation_metric == "hm":
                    # Validation
                    if best_hm is None or vhm > best_hm:
                        best_hm = vhm
                        # best_fc = concat_fc.clone()
                        # best_fc = concat_fc.detach().clone()
                        best_fc = deepcopy(self.fc.state_dict())
                        hm_patience_count = 0
                    else:
                        hm_patience_count += 1
                    lrc = scheduler.get_last_lr()[0]
                    out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (train) b/n:{:.3f}/{:.3f}, (test) b/n={:.3f}/{:.3f}, lrc={:.3f}'\
                        .format(
                            session, 
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0)),
                            float("%.3f" % (va * 100.0)),
                            float("%.3f" % (best_hm * 100.0)),
                            float("%.3f" % (taBase.item() * 100.0)),
                            float("%.3f" % (taNovel.item() * 100.0)),
                            float("%.3f" % (vaBase * 100.0)),
                            float("%.3f" % (vaNovel * 100.0)),
                            float(lrc))

                tqdm_gen.set_description(out_string)

                if hm_patience_count > hm_patience:
                    # faster joint session
                    break

                scheduler.step()
        
        self.fc.load_state_dict(best_fc, strict=True)    
    
    def update_fc_ft_joint_gaussian_mixup(self, jointloader, testloader, class_list, session, generated):
        gaus, aug_gaus = generated
        gaussian_data, gaussian_labels = gaus
        aug_gaus_data, aug_gaus_labels = aug_gaus

        # optimized_parameters = [base_fc, new_fc]
        if self.args.train_novel_in_joint:
            self.fc.novel_requires_grad(value=True)

        optimizer = self.get_optimizer_new(self.fc.parameters())

        criterion = nn.CrossEntropyLoss(label_smoothing = self.args.label_smoothing_joint)

        best_loss = None
        best_hm = None
        best_acc = None

        best_fc = None
        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        hm_patience = 15
        hm_patience_count = 0

        # shuffle generated data
        shuffler = np.arange(gaussian_data.shape[0])
        np.random.shuffle(shuffler)
        gaussian_data = gaussian_data[shuffler]
        gaussian_labels = gaussian_labels[shuffler]
        
        # split equally
        total_iter = jointloader.__len__()
        generated_data = np.array_split(gaussian_data, total_iter)
        generated_labels = np.array_split(gaussian_labels, total_iter)

        novel_class_start = self.args.base_class + (self.args.way * (session-1))

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))

            # To emulate augmentation in generated data we need to mixup the generated data
            # Basically a pair of samples from a single class will be averaged to create a new instance, the choice of pair is entirely random.

            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                # for i, batch in enumerate(tqdm_gen, 1):
                for i, (data, label) in enumerate(jointloader):

                    data = data.cuda()
                    label = label.cuda()
                    label[label >= self.args.base_class] -= label_offset

                    if self.args.mixup_joint:
                        data, targets_a, targets_b, lam = mixup_data(data, label, alpha = self.args.mixup_alpha)
                        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

                    encoding = self.encode(data).detach() #<<< The encoder is essentially frozen

                    # logits = self.get_logits(encoding, concat_fc)
                    logits_novel = self.fc.get_logits(encoding)
                    loss_novel = criterion(logits_novel, label) # Note, no label smoothing here

                    # From the gaussian generated data we now create base logits. but before that we must mixup the embeddings
                    # augmented_data = gaussian_utils.augmentMultivariateData(generated_data[i], generated_labels[i], aug_gaus_data, aug_gaus_labels)
                    augmented_data = torch.tensor(generated_data[i]).cuda().to(torch.float)
                    augmented_label = torch.tensor(generated_labels[i]).cuda().to(torch.long)
                    mixup_embedding_base, targets_a, targets_b, lam = mixup_data(augmented_data, augmented_label)
                    logits_base = self.fc.get_logits(mixup_embedding_base)
                    loss_base = mixup_criterion(criterion, logits_base, targets_a, targets_b, lam)

                    # Combine losses
                    loss = (loss_novel + loss_base) / 2.0
                    
                    ta.add(count_acc(logits_novel, label))
                    ta.add(count_acc(logits_base, augmented_label))

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), opt.clip)
                    optimizer.step()

                    total_loss += loss.item()

                # Model Saving
                vl, va, vaNovel, vaBase, vhm, vam, vbin = self.test_fc_head(self.fc, testloader, epoch, session)
                if self.args.validation_metric == "hm":
                    # Validation
                    if best_hm is None or vhm > best_hm:
                        best_hm = vhm
                        # best_fc = concat_fc.clone()
                        # best_fc = concat_fc.detach().clone()
                        best_fc = deepcopy(self.fc.state_dict())
                        hm_patience_count = 0
                    else:
                        hm_patience_count += 1
                    out_string = '(Joint) Sess: {}, loss {:.3f}, trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, b/n={:.3f}/{:.3f}'\
                        .format(
                            session, 
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0)),
                            float("%.3f" % (va * 100.0)),
                            float("%.3f" % (best_hm * 100.0)),
                            float("%.3f" % (vaBase * 100.0)),
                            float("%.3f" % (vaNovel * 100.0)),)

                tqdm_gen.set_description(out_string)

                # self.writer.add_scalars(f"(Joint) Session {session} Training Graph", {
                #     "jta": ta.item(),
                #     "jva": va,
                #     # "jvaNovel": vaNovel
                # }, epoch)

                if hm_patience_count > hm_patience:
                    # faster joint session
                    break
        
        self.fc.load_state_dict(best_fc, strict=True)    
    
    def test_fc_head(self, fc, testloader, epoch, session):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        vl = Averager()
        va = Averager()

        # >>> Addition
        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only
        vaBinary = Averager() # Averager for binary classification of novel and base classes


        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        # test_fc = fc.clone().detach()
        self.eval()

        with torch.no_grad():
            # tqdm_gen = tqdm(testloader)
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]
                test_label[test_label >= self.args.base_class] -= label_offset

                encoding = self.encode(data).detach()

                # logits = self.get_logits(encoding, fc)
                logits = fc.get_logits(encoding)
                # TODO: Instead of just taking the dot product during testing
                # We need to integrate the dot product as well as the l2 distance between the prototype and the feature extractor output

                logits = logits[:, :test_class]

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                bin_acc = count_acc_binary(logits, test_label, test_class, self.args)

                # >>> Addition
                novelAcc, baseAcc = count_acc_(logits, test_label, test_class, self.args)

                vaNovel.add(novelAcc)
                vaBase.add(baseAcc)
                vaBinary.add(bin_acc)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

            # >>> Addition 
            vaNovel = vaNovel.item()
            vaBase = vaBase.item()
            vaBinary = vaBinary.item()

        vhm = hm(vaNovel, vaBase)
        vam = am(vaNovel, vaBase)

        return vl, va, vaNovel, vaBase, vhm, vam, vaBinary

    def test_fc(self, fc, testloader, epoch, session, norm_encoding = False):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        vl = Averager()
        va = Averager()

        # >>> Addition
        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only
        vaBinary = Averager() # Averager for binary classification of novel and base classes

        label_offset = 0
        if self.args.base_only:
            # Remove the logits between the novel and base classes
            # or completely 0 them out.
            label_offset = (session - 1) * self.args.way

        # test_fc = fc.clone().detach()
        self.eval()

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for batch in tqdm_gen:
                data, test_label = [_.cuda() for _ in batch]
                test_label[test_label >= self.args.base_class] -= label_offset

                encoding = self.encode(data).detach()

                if self.args.incremental_on_supcon_head:
                    encoding = self.sup_con_head(encoding).detach()

                # TODO: Normalise encoding
                if norm_encoding:
                    encoding = F.normalize(encoding, dim=-1)

                logits = self.get_logits(encoding, fc)
                # logits = fc.get_logits(encoding)
                # TODO: Instead of just taking the dot product during testing
                # We need to integrate the dot product as well as the l2 distance between the prototype and the feature extractor output

                logits = logits[:, :test_class]

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                bin_acc = count_acc_binary(logits, test_label, test_class, self.args)

                # >>> Addition
                novelAcc, baseAcc = count_acc_(logits, test_label, test_class, self.args)

                vaNovel.add(novelAcc)
                vaBase.add(baseAcc)
                vaBinary.add(bin_acc)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

            # >>> Addition 
            vaNovel = vaNovel.item()
            vaBase = vaBase.item()
            vaBinary = vaBinary.item()

        vhm = hm(vaNovel, vaBase)
        vam = am(vaNovel, vaBase)

        return vl, va, vaNovel, vaBase, vhm, vam, vaBinary