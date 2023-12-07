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
import supcon

from tqdm import tqdm
from pdb import set_trace as bp

from utils import *
from scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR

from copy import deepcopy

from torch.autograd import Variable

from geom_median.torch import compute_geometric_median

import gaussian_utils

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

from torchvision.models import resnet18 as tv_resnet18

def compute_angles(vectors):
    # Compute angles for each vector with all others from 
    # Compute angles with the closesnt neighbour only
    proto = vectors.cpu().numpy()
    dot = np.matmul(proto, proto.T)
    dot = dot.clip(min=0, max=1)
    theta = np.arccos(dot)
    np.fill_diagonal(theta, np.nan)
    theta = theta[~np.isnan(theta)].reshape(theta.shape[0], theta.shape[1] - 1)
    
    avg_angle_close = theta.min(axis = 1).mean()
    avg_angle = theta.mean()

    return np.rad2deg(avg_angle), np.rad2deg(avg_angle_close)

class SRHead(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        
        self.args = args

        # Same as proj_output_dim
        self.num_features = num_features

        # Classifier for the base classes
        # Note: To create the base fc we need for each class the mean projection output
        self.base_fc = nn.Linear(self.num_features, self.args.base_class, bias=False)       # Note the entire number of classes are already added

        # Set of all classifiers
        self.classifiers = nn.Sequential(self.base_fc)

        # Register buffer for the rv
        self.num_classes = args.num_classes
        self.n_inc_classes = args.num_classes - args.base_class

        self.reserve_vector_count = self.args.reserve_vector_count
        if self.reserve_vector_count == -1:
            if self.args.reserve_mode in ["novel"]:
                self.reserve_vector_count = self.n_inc_classes
            elif self.args.reserve_mode in ["all", "base_init", "two_step"]:
                self.reserve_vector_count = self.num_classes

        self.register_buffer("rv", torch.randn(self.reserve_vector_count, self.num_features))

        self.radius = 1.0

        self.novel_temperature = 1.0
        self.base_temperature = 1.0

    def get_assignment(self, cost, assignment_mode, prototypes):
        """ Take cost array with cosine scores and return the output col ind """
        if assignment_mode == "max":
            row_ind, col_ind = linear_sum_assignment(cost, maximize = True)
        elif assignment_mode == "min":
            row_ind, col_ind = linear_sum_assignment(cost, maximize = False)
        elif assignment_mode == "random":
            col_ind = torch.randperm(self.rv.shape[0])[:cost.shape[0]]
        elif assignment_mode == "cosine_penalty":
            assigned_cost = compute_off_element_mean(cost)
            cost = (cost + 2*assigned_cost)
            row_ind, col_ind = linear_sum_assignment(cost, maximize = True)
        return col_ind

    def get_classifier_weights(self):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            output.append(cls.weight.data)
        return torch.cat(output, axis = 0)

    def assign_base_classifier(self, base_prototypes):
        # Normalise incoming prototypes
        base_prototypes = normalize(base_prototypes)

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(base_prototypes.cpu(), self.rv.cpu())
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(base_prototypes.cpu(), self.rv.cpu())
        # row_ind, col_ind = linear_sum_assignment(cost, maximize = True)

        col_ind = self.get_assignment(cost, self.args.assignment_mode_base, base_prototypes)
        
        new_fc_tensor = self.rv[col_ind]

        avg_angle, avg_angle_close = compute_angles(new_fc_tensor)
        print(f"Selected Base Classifiers have average angle: {avg_angle} and average closest angle: {avg_angle_close}")

        # Create fixed linear layer
        self.classifiers[0].weight.data = new_fc_tensor

        if not self.args.online_assignment:
            # Remove from the final rv
            all_idx = np.arange(self.rv.shape[0])
            self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]

    def assign_novel_classifier(self, new_prototypes, online = False):  
        # Normalise incoming prototypes
        new_prototypes = normalize(new_prototypes)

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(new_prototypes.cpu(), self.rv.cpu())
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(new_prototypes.cpu(), self.rv.cpu())
        # The linear sum assignment is maximised
        # row_ind, col_ind = linear_sum_assignment(cost, maximize = True)

        col_ind = self.get_assignment(cost, self.args.assignment_mode_novel, new_prototypes)
        
        new_fc_tensor = self.rv[col_ind]

        avg_angle, avg_angle_close = compute_angles(new_fc_tensor)
        print(f"Selected Novel Classifiers have average angle: {avg_angle} and average closest angle: {avg_angle_close}")

        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0], bias=False).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        self.classifiers.append(new_fc.cuda())

        # Now for each row_ind append the new rv and remove it from the final rv
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]

    def assign_online(self, prototypes):
        # Normalise incoming prototypes
        prototypes = normalize(prototypes)

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(prototypes.cpu(), self.rv.cpu())
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(prototypes.cpu(), self.rv.cpu())

        # Inherit assignment mode from base
        assignment_mode = self.args.assignment_mode_base
        col_ind = self.get_assignment(cost, assignment_mode, prototypes)
        
        fc_tensor = self.rv[col_ind]

        avg_angle, avg_angle_close = compute_angles(fc_tensor)
        print(f"Selected Base Classifiers have average angle: {avg_angle} and average closest angle: {avg_angle_close}")

        # Assume all classifier are already made, we now assign each classifier its tensor
        for i, cls in enumerate(self.classifiers.children()):
            if i == 0:
                cls.weight.data = fc_tensor[:self.args.base_class]
            else:
                n_class_start = self.args.base_class + (self.args.way * (i-1))
                n_class_end = self.args.base_class + (self.args.way * i)
                cls.weight.data = fc_tensor[n_class_start:n_class_end]

        # And rv remains un altered

    def remove_assigned_rv(self, col_ind):
        # Now for each row_ind append the new rv and remove it from the final rv
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]

    def find_reseverve_vectors(self):
        self.radius = 1.0
        self.temperature = 0.5
        
        base_prototypes = normalize(self.base_fc.weight.data)
        points = torch.randn(self.n_inc_classes, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=5)
        
        best_angle = 0
        tqdm_gen = tqdm(range(2000))

        for _ in tqdm_gen:
            # Combining prototypes but only optimising the reserve vector
            comb = torch.cat((points, base_prototypes), axis = 0)

            # Compute the cosine similarity.
            sim = F.cosine_similarity(comb[None,:,:], comb[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / comb.shape[0]
            
            # opt.zero_grad()
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(torch.cat((points, base_prototypes), axis = 0).detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data

    def find_reseverve_vectors_all(self):
        self.temperature = 1.0
        
        points = torch.randn(self.reserve_vector_count, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        
        best_angle = 0
        tqdm_gen = tqdm(range(self.args.epochs_simplex))

        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            
            # opt.zero_grad()
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data
        # self.register_buffer('rv', points.data)

    def find_reseverve_vectors_two_step(self, proto):
        self.temperature = 1.0
        
        proto = normalize(proto)
        points = torch.randn(self.n_inc_classes, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        best_angle = 0
        tqdm_gen = tqdm(range(1000))
        print("(Simplex search) Optimising the randn to be far away from base prototype")
        for _ in tqdm_gen:
            comb = torch.cat((proto, points), axis = 0)
            # Compute the cosine similarity.
            sim = F.cosine_similarity(comb[None,:,:], comb[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / comb.shape[0]
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle
            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # proto = torch.nn.Parameter(proto)
        # points = torch.cat((proto, points), axis = 0)
        points = torch.randn(self.num_classes, self.num_features).cuda()
        points.data = torch.cat((proto, points), axis = 0)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        tqdm_gen = tqdm(range(10000))
        print("(Simplex search) Optimising everything together")
        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle
            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data
    
    def find_reseverve_vectors_base_init(self, proto):
        self.temperature = 1.0
        
        proto = normalize(proto)
        points = torch.randn(self.reserve_vector_count - proto.shape[0], self.num_features).cuda()
        points = normalize(points)
        points = torch.cat((proto, points), axis = 0)
        points = torch.nn.Parameter(points)
        opt = torch.optim.SGD([points], lr=1)
        best_angle = 0
        tqdm_gen = tqdm(range(1000))
        print("(Simplex search) Optimising combined base+rand to be far away")
        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data
    
    def novel_requires_grad(self, value = True):
        for i, cls in enumerate(self.classifiers.children()):
            if isinstance(cls, nn.Sequential):
                cls[-1].weight.requires_grad = value
            elif isinstance(cls, nn.Sequential):
                cls.weight.requires_grad = value

    def forward(self, x):
        return self.get_logits(x)
    
    def append_novel_classifier(self, new_head):
        self.classifiers.append(new_head.cuda())

    def create_new_classifier(self, init=None):
        new_fc = nn.Linear(self.num_features, self.args.way, bias=False).cuda()
        if init is not None:
            new_fc.weight.data.copy_(init)

        self.classifiers.append(new_fc)
        
    def get_logits(self, encoding, session = 0):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            out = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            temp = self.base_temperature if i == 0 else self.novel_temperature

            out = out / temp

            # Note: No detaching required given that the the the classifiers do not update
            output.append(out)
        output = torch.cat(output, axis = 1)
        return output

    def get_dot(self, encoding, session = 0):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            out = cls(encoding)
            temp = self.base_temperature if i == 0 else self.novel_temperature

            out = out / temp

            # Note: No detaching required given that the the the classifiers do not update
            output.append(out)
        output = torch.cat(output, axis = 1)
        return output

        
class MYNET(nn.Module):

    def __init__(self, args, mode=None, writer=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            # == Alice resnet
            # self.encoder = resnet_CIFAR.ResNet18()

            # == Basic torchvision resnet
            self.encoder = tv_resnet18()
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.encoder.maxpool = nn.Identity()

            self.encoder.fc = nn.Identity()
            self.encoder_outdim = 512
            self.proj_hidden_dim = 2048
            self.proj_output_dim = 128
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet_CIFAR.ResNet18()  # out_dim = 128
            self.encoder.fc = nn.Identity()
            self.encoder_outdim = 512

            # From solo learn
            self.proj_hidden_dim = self.args.proj_hidden_dim #2048
            self.proj_output_dim = self.args.proj_output_dim #128
        if self.args.dataset == 'cub200':
            # Original 
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.encoder_outdim = 512
            self.proj_hidden_dim = 2048
            self.proj_output_dim = 256

            # Torch vision Resnet
            # self.encoder = tv_resnet18()
            # self.encoder.fc = nn.Identity()
            
            # self.encoder_outdim = 512
            # self.proj_hidden_dim = 2048
            # self.proj_output_dim = 256

        self.writer = writer

        # Sup con projection also the projector which gets fine tuned during the joint session
        self.projector = self.select_projector()

        # Note. FC is created from the mean outputs of the final linear layer of the projector for all class samples
        self.fc = SRHead(self.args, self.proj_output_dim)

        # For hard posiitves in the training set
        self.path2conf = {}

    def select_projector(self):
        if self.args.proj_type == "linear":
            projector = nn.Linear(self.encoder_outdim, self.proj_output_dim)
        elif self.args.proj_type == "proj":
            # projector
            projector = nn.Sequential(
                nn.Linear(self.encoder_outdim, self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            )
        elif self.args.proj_type == "proj_alice":
            projector = nn.Sequential(
                nn.Linear(self.encoder_outdim, self.proj_hidden_dim),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            )
        elif self.args.proj_type == "proj_ncfscil":
            projector = nn.Sequential(
                nn.Linear(self.encoder_outdim, self.encoder_outdim * 2),
                nn.BatchNorm1d(self.encoder_outdim * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(self.encoder_outdim * 2, self.encoder_outdim * 2),
                nn.BatchNorm1d(self.encoder_outdim * 2),
                nn.LeakyReLU(0.1),
                nn.Linear(self.encoder_outdim * 2, self.proj_output_dim, bias=False),
            )
        return projector

    def forward_sup_con(self, x, **kwargs):
        if self.args.skip_encode_norm:
            x = self.encode(x)
        else:  
            x = F.normalize(self.encode(x), dim = 1)

        if self.args.skip_sup_con_head:
            return x
        
        x = F.normalize(self.sup_con_head(x), dim=1)
        
        return x

    def init_proj_random(self):
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_outdim, self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        ).cuda()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.projector.apply(init_weights)
        
    def forward_metric(self, x):
        # Get projection output
        g_x = self.encode(x)

        # Get similarity scores between classifier prototypes and g_x
        sim = self.fc.get_logits(g_x, 0)

        return sim, g_x

    def encode(self, x, detach_f = False):
        x = self.encoder(x)

        # Only finetuning the projector
        if detach_f: x = x.detach()

        x = self.projector(x)

        # Note the output is unnormalised
        return x

    def forward(self, input, **kwargs):
        if self.mode == "backbone":  # Pass only through the backbone
            input = self.encoder(input)
            return input
        if self.mode == 'encoder':
            input = self.encode(input, **kwargs)
            return input
        elif self.mode not in ['encoder', "backbone"]:
            input, encoding = self.forward_metric(input)
            return input, encoding
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, trainloader, testloader, class_list, session, mode="encoder"):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            if mode=="encoder":
                data=self.encode(data).detach()
            elif mode == "backbone":
                data=self.encoder(data).detach()

        new_prototypes = self.get_class_avg(data, label, class_list)
        
        self.fc.create_new_classifier(init=new_prototypes)

        # Use bnce to fine tune novel classifier
        if self.args.apply_bnce:
            self.update_fc_ft_novel(trainloader, testloader, session)

        return new_prototypes

    def update_fc_online(self, trainloader, class_list):
        # To handle batched data
        all_data = []
        all_label = []
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()
            all_data.append(data)
            all_label.append(label)
        all_data = torch.cat(all_data, axis = 0)
        all_label = torch.cat(all_label)
 
        online = self.get_class_avg(all_data, all_label, class_list)

        self.fc.create_new_classifier() # Create a novel classifier
        print("===Creating new classifier")
        self.fc.assign_online(online)   # Assign to all classes
        print("===Assigned best target to all classifier")

    def get_class_avg(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            new_fc.append(proto)

        new_fc_tensor=torch.stack(new_fc,dim=0)

        return new_fc_tensor

    def get_logits(self,x,fc):
        return F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)) # self.args.temperature

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9 if not self.args.nesterov_new else 0, weight_decay=self.args.decay_new, nesterov=self.args.nesterov_new)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)
        return optimizer

    def update_fc_ft_novel(self, trainloader, testloader, session):
        theta1 = deepcopy(self.projector.state_dict())
        kp_lam = self.args.kp_lam

        optimizer = torch.optim.SGD([
            # {'params': self.projector.parameters()},
            {'params': self.fc.classifiers[1:].parameters()}
        ], lr=0.005, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)

        # criterion = nn.CrossEntropyLoss()
        criterion = self.select_criterion()

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],
                                                             gamma=self.args.gamma)

        best_loss = None
        best_acc = None
        best_hm = None

        self.eval() # Fixing batch norm

        average_gap = Averager()
        average_gap_n = Averager()
        average_gap_b = Averager()

        # Before training accuracies
        val_freq = self.args.epochs_novel

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_novel))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                for data, label in trainloader:
                    data = data.cuda()
                    label = label.cuda()

                    if self.args.instance_mixup:
                        data, label = instance_mixup_data(data, label)
                        data, label = map(Variable, (data, label))

                    encoding = self.encode(data, detach_f=True)

                    # Get the cosine similarity to the classifier
                    # logits = self.fc(encoding)
                    logits = self.fc.get_dot(encoding)

                    # Take the maximum logit score in the novel classifier and the maximum logit score in the previous sets of classes find the different in the logit outputs. 
                    # Compute this distance for each batch sample and print
                    for i in range(logits.shape[0]):
                        average_gap.add(logits[i][self.args.base_class + (self.args.way * (session-1)):].max() - logits[i][:self.args.base_class].max())

                    novel_logits = logits[:, self.args.base_class + (self.args.way * (session-1)):]
                    base_logits = logits[:, :self.args.base_class]
                    average_gap_n.add((novel_logits.max(axis=1)[0] - novel_logits.min(axis=1)[0]).mean())
                    average_gap_b.add((base_logits.max(axis=1)[0] - base_logits.min(axis=1)[0]).mean())

                    # BNCE Loss
                    loss = self.criterion_forward(criterion, logits, label)
                    # loss += KP_loss(theta1, self.projector, lymbda_kp=kp_lam)
                    
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
                        
                tqdm_gen.set_description(out_string)

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
        # Optimising only the projector
        optimizer = self.get_optimizer_new(self.projector.parameters())

        # Setting up the scheduler
        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)
        else:
            warmup_epochs = 10
            min_lr = 0.0
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=10, max_epochs=self.args.epochs_joint,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                    eta_min=min_lr)

        criterion = self.select_criterion()
        xent = nn.CrossEntropyLoss()

        best_loss = None
        best_hm = None
        best_acc = None
        best_projector = None

        hm_patience = self.args.hm_patience
        hm_patience_count = 0
        
        average_gap = Averager()
        average_gap_n = Averager()
        average_gap_b = Averager()
        
        if self.args.novel_bias:
            novel_class_start = self.args.base_class + (self.args.way * (session-1))
        else:
            novel_class_start = self.args.base_class

        self.eval() # Fixing batch norm

        test_freq = self.args.testing_freq

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                tl = {
                    "novel": Averager(),
                    "base": Averager()
                }
                for data, label in jointloader:
                    data = data.cuda()
                    label = label.cuda()

                    if self.args.instance_mixup:
                        data, label = instance_mixup_data(data, label)
                        data, label = map(Variable, (data, label))

                    encoding = self.encode(data, detach_f=True)

                    # Get the cosine similarity to the classifier
                    logits = self.fc(encoding)

                    # Compute Average Logit Gaps
                    novel_targets = label >= (self.args.base_class + (self.args.way * (session-1)))
                    if True in novel_targets:
                        novel_logits = logits[novel_targets, self.args.base_class + (self.args.way * (session-1)):]
                        base_logits = logits[novel_targets, :self.args.base_class]

                        average_gap.add((novel_logits.max(axis=1)[0] - base_logits.max(axis=1)[0]).mean())
                        average_gap_n.add((novel_logits.mean(axis = 1)).mean())  # Computes the mean of logits/cosine scores across each logit
                        average_gap_b.add((base_logits.mean(axis = 1)).mean())
                    
                    if self.args.mixup_joint:
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        if self.args.joint_loss in ['ce_even', 'ce_inter']:
                            losses = []
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            # novel_classes_idx = torch.argwhere(label >= self.args.base_class).flatten()
                            if self.args.joint_loss == "ce_even":
                                base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                                # base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()
                            elif self.args.joint_loss == "ce_inter":
                                inter_classes_idx = torch.argwhere((label >= self.args.base_class) & (label < novel_class_start)).flatten()
                                base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()

                                if inter_classes_idx.numel() != 0:
                                    inter_loss = criterion(logits[inter_classes_idx, :], label[inter_classes_idx])
                                    losses.append(inter_loss)

                            if novel_classes_idx.numel() != 0:
                                # Loss computed using the novel classes
                                novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label[novel_classes_idx])
                                tl["novel"].add(novel_loss.item())
                                losses.append(novel_loss)

                            if base_classes_idx.numel() != 0:
                                base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label[base_classes_idx])
                                # base_loss /= base_classes_idx.numel()
                                tl["base"].add(base_loss.item())
                                losses.append(base_loss)

                            loss = 0
                            loss_string = f"Losses =>"
                            idx2name = {0:"novel", 1:"base"}
                            for idx, l in enumerate(losses): 
                                loss += l
                                loss_string += f" loss {idx2name[idx]}: {loss:.3f},"
                            # tqdm_gen.set_description(loss_string)
                            loss /= len(losses)

                            loss += self.args.xent_weight * xent(logits, label)

                            # Use the cosine push loss to push hard negatives
                            # loss += compute_hn_cosine_embedding_loss(encoding, logits, label, session)
                        elif self.args.joint_loss == "ce_weighted":
                            novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                            base_classes_idx = torch.argwhere(label < novel_class_start).flatten()

                            # w_current = (1/(session+1))     # 0.5, 0.333, 0.25
                            # w_replay = 1-w_current          # 0.5, 0.667, ...
                            w_replay = (1/(session+1))     # 0.5, 0.333, 0.25
                            w_current = 1-w_replay          # 0.5, 0.667, ...

                            loss = 0
                            if novel_classes_idx.numel() != 0:
                                novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label[novel_classes_idx])
                                loss += w_current * novel_loss

                            if base_classes_idx.numel() != 0:
                                base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label[base_classes_idx])
                                loss += w_replay * base_loss
                        else:
                            loss = criterion(logits, label)

                    ta.add(count_acc(logits, label))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # == Deprecated ==
                # vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)

                # Model Saving
                if epoch % test_freq == 0:
                    metrics = self.test_fc_head(self.fc, testloader, epoch, session)
                    if self.args.validation_metric == "hm":
                        # Validation
                        vhm = metrics["vhm"]
                        if best_hm is None or vhm > best_hm:
                            best_hm = vhm
                            hm_patience_count = -1
                            best_projector = deepcopy(self.projector.state_dict())
                        out_string = '(Joint) Sess: {}, loss {:.3f}|(b/n)({:.3f}/{:.3f}), trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (test) b/n={:.3f}/{:.3f}, bb/nn={:.3f}/{:.3f}'\
                            .format(
                                    session, 
                                    total_loss,
                                    float('%.3f' % (tl["base"].item())),
                                    float('%.3f' % (tl["novel"].item())),
                                    float('%.3f' % (ta.item() * 100.0)),
                                    float("%.3f" % (metrics["va"] * 100.0)),
                                    float("%.3f" % (best_hm * 100.0)),
                                    float("%.3f" % (metrics["vaBase"] * 100.0)),
                                    float("%.3f" % (metrics["vaNovel"] * 100.0)),
                                    float("%.3f" % (metrics["vaBaseB"] * 100.0)),
                                    float("%.3f" % (metrics["vaNovelN"] * 100.0)),
                                    # float('%.3f' % (average_gap.item())),
                                    # float('%.3f' % (average_gap_b.item())),
                                    # float('%.3f' % (average_gap_n.item())),
                                )
                        fp = metrics["fpr"]
                        out_string += f", (False Positives) n2b|b2n|n2n|b2b:{fp['novel2base']:.3f}|{fp['base2novel']:.3f}|{fp['novel2novel']:.3f}|{fp['base2base']:.3f}"
                        hm_patience_count += 1

                tqdm_gen.set_description(out_string)

                if hm_patience_count > hm_patience:
                    break
                
                scheduler.step()
        
        self.projector.load_state_dict(best_projector, strict=True)
    
    def get_hard_base_classes(self, trainset, trainloader):
        """
            Get the testing score for the fc that is being currently trained
        """
        trainset.get_ix = True
        self.eval()        
        with torch.no_grad():
            # tqdm_gen = tqdm(trainloader)
            for i, pair in enumerate(tqdm(trainloader)):
                data = pair[0].cuda()
                label = pair[1].cuda()
                ixs = pair[2].cuda()

                encoding = self.encode(data).detach()
                logits = self.fc.get_logits(encoding)

                for j, ix in enumerate(ixs):
                    path = trainset.data[ix]
                    conf = logits[j, label[j]].item()
                    class_id = label[j].item()

                    if class_id not in self.path2conf:
                        self.path2conf[class_id] = {"path":[], "conf":[]}
                    self.path2conf[class_id]["path"].append(path)
                    self.path2conf[class_id]["conf"].append(conf)

    def test_fc_head(self, fc, testloader, epoch, session):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        vl = Averager()
        va = Averager()

        fpr = {
            "novel2base": 0,    # Count of samples from novel classes selected as base / Total novel samples
            "base2novel": 0,    # Count of samples from base classes selected as novel 
            "base2base": 0,     # Count of samples from base classes selected as other base
            "novel2novel": 0,    # Count of samples from novel classes selected as other novels
            "total_novel": 0,
            "total_base": 0
        }

        # >>> Addition
        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only
        vaBinary = Averager() # Averager for binary classification of novel and base classes
        vaNovelN = Averager() # Averager for novel classes only
        vaBaseB = Averager() # Averager for binary classification of novel and base classes

        # test_fc = fc.clone().detach()
        self.eval()

        total_novel_samples = 0
        total_base_samples = 0

        with torch.no_grad():
            # tqdm_gen = tqdm(testloader)
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]

                encoding = self.encode(data).detach()

                logits = fc.get_logits(encoding)

                logits = logits[:, :test_class]

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                bin_acc = count_acc_binary(logits, test_label, test_class, self.args)

                # >>> Addition
                novelAcc, baseAcc = count_acc_(logits, test_label, test_class, self.args)
                if session > 0:
                    fpr = count_fp(logits, test_label, test_class, self.args, fpr)
                    novelNAcc, baseBAcc = count_acc_(logits, test_label, test_class, self.args, sub_space="separate")
                    vaNovelN.add(novelNAcc)
                    vaBaseB.add(baseBAcc)

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
            vaNovelN = vaNovelN.item()
            vaBaseB = vaBaseB.item()

        vhm = hm(vaNovel, vaBase)
        vam = am(vaNovel, vaBase)

        if session > 0:
            fpr["novel2base"] /= fpr["total_novel"]
            fpr["base2novel"] /= fpr["total_base"]
            fpr["novel2novel"] /= fpr["total_novel"]
            fpr["base2base"] /= fpr["total_base"]

        metrics = {
            "vl":  vl,
            "va": va,
            "vaNovel" : vaNovel,
            "vaBase" : vaBase,
            "vhm" : vhm,
            "vam" : vam,
            "vaBinary" : vaBinary,
            "vaNovelN" : vaNovelN,
            "vaBaseB": vaBaseB,
            "fpr": fpr
        }

        return metrics

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

                # Get encoding
                encoding = self.encode(data).detach()
                logits = self.get_logits(encoding, fc)
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

    def test_backbone(self, fc, testloader, epoch, session):
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

        # test_fc = fc.clone().detach()
        self.eval()

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for batch in tqdm_gen:
                data, test_label = [_.cuda() for _ in batch]

                # Get encoding
                encoding = self.encoder(data).detach()
                # logits = F.linear(encoding, fc)
                logits = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

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

    def select_criterion(self):
        if self.args.criterion == "xent":
            return nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        elif self.args.criterion == "cosine":
            return torch.nn.CosineEmbeddingLoss()
        elif self.args.criterion == "xent+cosine":
            return [nn.CrossEntropyLoss(), torch.nn.CosineEmbeddingLoss()]
        elif self.args.criterion == "none":
            return None

    def criterion_forward(self, criterion, logits, label):
        if self.args.criterion == "xent":
            return criterion(logits, label)
        elif self.args.criterion == "cosine":
            target = torch.ones_like(label)
            one_hot_label = F.one_hot(label, logits.shape[1])
            return radial_label_smoothing(criterion, logits, one_hot_label, target, self.args.radial_label_smoothing)
        elif self.args.criterion == "xent+cosine":
            xent_loss = criterion[0](logits, label)
            target = torch.ones_like(label)
            one_hot_label = F.one_hot(label, logits.shape[1])
            cosine_loss = radial_label_smoothing(criterion[1], logits, one_hot_label, target, self.args.radial_label_smoothing)
            return cosine_loss + self.args.xent_weight*xent_loss
        elif self.args.criterion == "cosine-squared":
            # https://arxiv.org/pdf/1901.10514.pdf
            # loss = torch.sum(torch.pow(1 - logits[torch.arange(label.shape[0]), label], 2))
            # import pdb; pdb.set_trace()
            loss = (1 - logits[torch.arange(label.shape[0]), label]).pow(2).sum()
            return loss
        elif self.args.criterion == "none":
            return 0
        
    def update_fc_ft_base(self, baseloader, testloader):
        if self.args.skip_base_ft:
            return self.test_fc_head(self.fc, testloader, 0, 0)["va"]

        # Optimising only the projector
        optimizer = optimizer = torch.optim.SGD(self.projector.parameters(),lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)

        # Setting up the scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80],gamma=self.args.gamma)

        criterion = self.select_criterion()

        best_loss = None
        best_hm = None
        best_acc = 0
        best_projector = None

        self.eval() # Fixing batch norm

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_base))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                for idx, batch in enumerate(baseloader):
                    data, label = batch
                    data = data.cuda()
                    label = label.cuda()

                    if self.args.instance_mixup:
                        data, label = instance_mixup_data(data, label)
                        data, label = map(Variable, (data, label))

                    encoding = self.encode(data, detach_f=True)

                    logits = self.fc(encoding)

                    loss = self.criterion_forward(criterion, logits, label)

                    ta.add(count_acc(logits, label))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    out_string = f"Epoch: {epoch}|[{idx}/{len(baseloader)}], Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {best_acc*100:.3f}"
                    tqdm_gen.set_description(out_string)

                # Model Saving
                test_out = self.test_fc_head(self.fc, testloader, epoch, 0)
                va = test_out["va"]
                
                if best_acc is None or best_acc < va:
                    best_acc = va

                out_string = f"Epoch: {epoch}, Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {va*100:.3f}"
                tqdm_gen.set_description(out_string)
                
                scheduler.step()

        va = self.test_fc_head(self.fc, testloader, 0, 0)["va"]
        return va


    def update_fc_ft_joint_supcon(self, jointloader, testloader, class_list, session):
        optimizer = torch.optim.SGD([
            {'params': self.projector.parameters()},
            # {'params': self.fc.classifiers[1:].parameters()}
            {'params': self.fc.classifiers.parameters(), 'lr':self.args.lr_new}
        ], lr=self.args.lr_new/0.1, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)


        # Setting up the scheduler
        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120],gamma=self.args.gamma)
        else:
            warmup_epochs = 10
            min_lr = 0.0
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=10, max_epochs=self.args.epochs_joint,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                    eta_min=min_lr)

        sc_criterion = supcon.SupConLoss(lam_at = self.args.lam_at)
        criterion = self.select_criterion()

        best_loss = None
        best_hm = None
        best_acc = None
        best_projector = None
        best_l4 = None
        best_hm_epoch = 0

        hm_patience = self.args.hm_patience
        hm_patience_count = 0
        
        average_gap = Averager()
        average_gap_n = Averager()
        average_gap_b = Averager()
        
        if self.args.novel_bias:
            novel_class_start = self.args.base_class + (self.args.way * (session-1))
        else:
            novel_class_start = self.args.base_class


        self.eval() # Fixing batch norm

        test_freq = self.args.testing_freq

        # metrics = self.test_fc_head(self.fc, testloader, 0, session)
        # print(f"before starting joint session VA: {metrics['va'] * 100.0:3f}")

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                tl = {
                    "novel": Averager(),
                    "base": Averager()
                }
                for batch in jointloader:
                    images, label = batch
                    images = torch.cat([images[0], images[1]], dim=0)
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        label = label.cuda(non_blocking=True)

                    bsz = label.shape[0]

                    encoding = self.encode(images, detach_f=True)

                    # Normalise encoding
                    encoding = normalize(encoding)
                    f1, f2 = torch.split(encoding, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    sc_loss = sc_criterion(features, label, target_prototypes = None)

                    # TODO: Test with removing the second view. Maybe the second view is increasing the batch size and hence finding local minimas faster
                    logits = self.fc(encoding)
                    label = label.repeat(2)
                    novel_classes_idx = torch.argwhere(label >= novel_class_start).flatten()
                    base_classes_idx = torch.argwhere(label < novel_class_start).flatten()
                    novel_loss = 0
                    base_loss = 0
                    if novel_classes_idx.numel() != 0:
                        novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label[novel_classes_idx])                            
                        tl["novel"].add(novel_loss.item())
                    if base_classes_idx.numel() != 0:
                        base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label[base_classes_idx])
                        tl["base"].add(base_loss.item())
                    xent_loss = (0.5*base_loss) + (0.5*novel_loss)
                    

                    ta.add(count_acc(logits, label))
                    loss = sc_loss + xent_loss

                    if self.args.compute_hardnegative:
                        loss += 1e-3*compute_hn_cosine_embedding_loss(encoding, logits, label, session)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # == Deprecated ==
                # vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)

                # Model Saving
                if epoch % test_freq == 0:
                    if self.args.validation_metric == "hm":
                        metrics = self.test_fc_head(self.fc, testloader, epoch, session)
                        # Validation
                        vhm = metrics["vhm"]
                        if best_hm is None or vhm > best_hm:
                            best_hm = vhm
                            best_hm_epoch = epoch
                            hm_patience_count = -1
                        out_string = '(Joint) Sess: {}, loss {:.3f}|(b/n)({:.3f}/{:.3f}), trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (test) b/n={:.3f}/{:.3f}, bb/nn={:.3f}/{:.3f}'\
                            .format(
                                    session, 
                                    total_loss,
                                    float('%.3f' % (tl["base"].item())),
                                    float('%.3f' % (tl["novel"].item())),
                                    float('%.3f' % (ta.item() * 100.0)),
                                    float("%.3f" % (metrics["va"] * 100.0)),
                                    float("%.3f" % (best_hm * 100.0)),
                                    float("%.3f" % (metrics["vaBase"] * 100.0)),
                                    float("%.3f" % (metrics["vaNovel"] * 100.0)),
                                    float("%.3f" % (metrics["vaBaseB"] * 100.0)),
                                    float("%.3f" % (metrics["vaNovelN"] * 100.0)),
                                )
                        fp = metrics["fpr"]
                        out_string += f", (False Positives) n2b|b2n|n2n|b2b:{fp['novel2base']:.3f}|{fp['base2novel']:.3f}|{fp['novel2novel']:.3f}|{fp['base2base']:.3f}"
                        hm_patience_count += 1
                    elif self.args.validation_metric == "none":
                        out_string = '(Joint) Sess: {}, loss {:.3f}|(b/n)({:.3f}/{:.3f})'\
                            .format(
                                    session, 
                                    total_loss,
                                    float('%.3f' % (tl["base"].item())),
                                    float('%.3f' % (tl["novel"].item())),
                                    float('%.3f' % (ta.item() * 100.0)),
                                )

                tqdm_gen.set_description(out_string)

                if hm_patience_count > hm_patience:
                    break
                
                scheduler.step()

        print("Best HM found at epoch: ", best_hm_epoch)
        
        # self.projector.load_state_dict(best_projector, strict=True)
        # if self.args.fine_tune_backbone_joint:
        #     self.encoder.layer4.load_state_dict(best_l4, strict = True)