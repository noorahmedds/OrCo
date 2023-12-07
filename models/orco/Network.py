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
from models.resnet12_nc import *

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

from ema_pytorch import EMA

import math

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
            elif self.args.reserve_mode in ["all", "base_init", "two_step", "etf", "identity"]:
                self.reserve_vector_count = self.num_classes
            elif self.args.reserve_mode in ["full"]:
                self.reserve_vector_count = self.num_features
            elif self.args.reserve_mode in ["half"]:
                self.reserve_vector_count = self.num_features//2
            elif self.args.reserve_mode in ["colinear_negatives"]:
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

    def get_classifier_weights(self, uptil = -1):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            if uptil >= 0 and uptil < i + 1:
                break
            output.append(cls.weight.data)
        return torch.cat(output, axis = 0)

    def assign_base_classifier(self, base_prototypes, base_cov=None):
        # Normalise incoming prototypes
        base_prototypes = normalize(base_prototypes)

        # TODO: Add option to sample only args.base_class
        target_choice_ix = self.reserve_vector_count
        if self.args.target_sampling:
            target_choice_ix = self.args.base_class

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "mahalanobis":
            cost = compute_pairwise_mahalanobis(base_prototypes.cpu(), base_cov, self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "cos_odd_inv":
            # Odd indexes are inversed
            cost = cosine_similarity(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
            inv_ix = np.arange(cost.shape[0])[1::2]
            cost[inv_ix] = 1/cost[inv_ix]

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

    def assign_novel_classifier(self, new_prototypes, online = False, cov_list = None):  
        # Normalise incoming prototypes
        new_prototypes = normalize(new_prototypes)

        target_choice_ix = self.reserve_vector_count
        if self.args.target_sampling:
            target_choice_ix = self.args.way

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(new_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(new_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "mahalanobis":
            cost = compute_pairwise_mahalanobis(new_prototypes.cpu(), cov_list, self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "cos_odd_inv":
            # Odd indexes are inversed
            cost = cosine_similarity(new_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
            inv_ix = np.arange(cost.shape[0])[1::2]
            cost[inv_ix] = 1/cost[inv_ix]

        # The linear sum assignment is maximised
        # row_ind, col_ind = linear_sum_assignment(cost, maximize = True)

        if self.args.assign_flip:
            self.args.assignment_mode_novel = "max" if self.args.assignment_mode_novel == "min" else "min" 
            
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
        
        if self.args.reserve_mode == "colinear_negatives":
            points = torch.randn(int(self.reserve_vector_count/2), self.num_features).cuda()
        elif self.args.reserve_mode == "etf":
            orth_vec = generate_random_orthogonal_matrix(self.num_features, self.reserve_vector_count).cuda()
            i_nc_nc = torch.eye(self.reserve_vector_count).cuda()
            one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.reserve_vector_count, self.reserve_vector_count), (1 / self.reserve_vector_count)).cuda()
            etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                                math.sqrt(self.reserve_vector_count / (self.reserve_vector_count - 1)))
            self.rv = etf_vec.T
            return
        elif self.args.reserve_mode == "identity":
            i_nc_nc = torch.eye(self.num_features).cuda()[:self.reserve_vector_count]
            self.rv = i_nc_nc
            return
        else:
            points = torch.randn(self.reserve_vector_count, self.num_features).cuda()

        points = normalize(points)
        points = torch.nn.Parameter(points)
        
        if self.args.skip_orth:
            self.rv = points.data
            return

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

        if self.args.reserve_mode == "colinear_negatives":
            # Find colinear negative which are basically just opposite to one of the axis that we found
            # They are pairwise orthogonal to every vector but this way in a single plan we have 4 points.
            self.rv = torch.cat([points.data, -1*points.data], axis = 0)
        else:
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
        if init is None:
            new_fc = nn.Linear(self.num_features, self.args.way, bias=False).cuda()

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
    
    def get_spread_logits(self, encoding, class_variance):
        output = []
        # Generate perturbations
        for i, cls in enumerate(self.classifiers.children()):
            perturbations = []
            num_features = cls.weight.shape[1]
            for k in range(cls.weight.shape[0]):
                if i == 0:
                    offset = 0
                else:
                    offset = self.args.base_class
                var_ix = k + offset
                # Perturb the class weights by gaussian generated from class variance
                x = torch.from_numpy(np.random.multivariate_normal(cls.weight[k, :].detach().cpu().numpy(),  np.eye(num_features)*class_variance[var_ix,:].T, 1))
                perturbations.append(x)
            perturbations = torch.cat(perturbations, axis=0).to(cls.weight.device).type(cls.weight.type())

            out = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(cls.weight + perturbations, p=2, dim=-1))
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
            if "FSCIL_ALICE" in self.args.model_dir:
                # == Alice resnet
                self.encoder = resnet_CIFAR.ResNet18()
                self.encoder_outdim = 512
                self.proj_hidden_dim = 2048
                self.proj_output_dim = 2048  #128
                self.encoder.fc = nn.Identity()
            else:
                # == Resnet12_nc
                self.encoder = resnet12_nc()
                self.encoder_outdim = 640
                self.proj_hidden_dim = 2048
                self.proj_output_dim = 128 if self.args.proj_output_dim < 0 else self.args.proj_output_dim # 128

            # == Basic torchvision resnet
            # self.encoder = tv_resnet18()
            # self.encoder.conv1 = nn.Conv2d(
            #     3, 64, kernel_size=3, stride=1, padding=2, bias=False
            # )
            # self.encoder.maxpool = nn.Identity()

            # self.encoder.fc = nn.Identity()
            # self.encoder_outdim = 512
            # self.proj_hidden_dim = 2048
            # self.proj_output_dim = 128
            
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet_CIFAR.ResNet18()  # out_dim = 128
            self.encoder.fc = nn.Identity()
            self.encoder_outdim = 512

            # From solo learn
            self.proj_hidden_dim = self.args.proj_hidden_dim #2048
            self.proj_output_dim = 128 if self.args.proj_output_dim < 0 else self.args.proj_output_dim # 128
        if self.args.dataset == 'cub200':
            # Original 
            # self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            # self.encoder_outdim = 512
            # self.proj_hidden_dim = 2048
            # self.proj_output_dim = 256

            # Torch vision Resnet
            # self.encoder = tv_resnet18()
            self.encoder = tv_resnet18(pretrained=True) # Originally we use this
            # self.encoder = tv_resnet18(weights = "ResNet18_Weights.DEFAULT")
            self.encoder.fc = nn.Identity()
            
            self.encoder_outdim = 512

            self.proj_hidden_dim = 2048
            self.proj_output_dim = 256 # 256

        self.writer = writer

        # Sup con projection also the projector which gets fine tuned during the joint session
        self.projector = self.select_projector()

        # Note. FC is created from the mean outputs of the final linear layer of the projector for all class samples
        self.fc = SRHead(self.args, self.proj_output_dim)

        # For hard posiitves in the training set
        self.path2conf = {}

        self.projector_ema = None

    def init_proj_ema(self):
        self.projector_ema = EMA(
            self.projector,
            beta = self.args.proj_ema_beta,
            update_after_step = 10,    
            update_every = self.args.proj_ema_update_every
        ).cuda()
        self.center = torch.zeros(1, self.proj_output_dim).cuda()
        self.teacher_temp = self.args.teacher_temp
        self.student_temp = self.args.student_temp

    def update_center(self, teacher_output, center_momentum=0.9):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        self.center = self.center*center_momentum + batch_center * (1 - center_momentum)

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
        elif self.args.proj_type == "proj_mlp":
            projector = projection_MLP(self.encoder_outdim, self.proj_output_dim, 2)

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

    def encode(self, x, detach_f = False, return_encodings = False):
        encodings = self.encoder(x)

        # Only finetuning the projector
        if detach_f: encodings = encodings.detach()

        x = self.projector(encodings)

        if return_encodings:
            return x, encodings

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
        if not self.args.online_assignment: # Skip assignment if online assignment
            for batch in trainloader:
                data, label = [_.cuda() for _ in batch]
                if mode=="encoder":
                    data=self.encode(data).detach()
                elif mode == "backbone":
                    data=self.encoder(data).detach()

            new_prototypes, cov_list = self.get_class_avg(data, label, class_list)

            # Assign a new novel classifier from the given reseve vectors
            # Out of these reserve vectors we choose vectors which minimize the shift of the projector
            # I.e. we choose reserve vectors for each class in the incremental session
            # Based on the linear sum assignment
            if mode == "encoder":
                self.fc.assign_novel_classifier(new_prototypes, cov_list=cov_list)
        else:
            print ("Novel assignment skipped. Performing Assignmnet in the joint session only")

        # Use bnce to fine tune novel classifier
        if self.args.apply_bnce:
            self.update_fc_ft_novel(trainloader, testloader, session)

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
 
        online, cov_list = self.get_class_avg(all_data, all_label, class_list)

        self.fc.create_new_classifier() # Create a novel classifier
        print("===Creating new classifier")
        self.fc.assign_online(online)   # Assign to all classes
        print("===Assigned best target to all classifier")

    def get_class_avg(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        cov_list = []
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]

            # Compute covariance matrix again
            cov_this = np.cov(normalize(embedding).cpu(), rowvar=False)
            cov_list.append(cov_this)

            proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            new_fc.append(proto)

        new_fc_tensor=torch.stack(new_fc,dim=0)

        return new_fc_tensor, cov_list

    def get_logits(self,x,fc):
        return F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)) # self.args.temperature

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer_joint == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9 if not self.args.nesterov_new else 0, weight_decay=self.args.decay_new, nesterov=self.args.nesterov_new)
        elif self.args.optimizer_joint == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)   # wd:1e-6
        elif self.args.optimizer_joint == "adamw":
            optimizer = torch.optim.AdamW(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)  # wd:1e-1
        return optimizer

    def get_optimizer_base(self, optimized_parameters):
        # # Optimising only the projector
        # if self.args.fine_tune_backbone_base:
        #     optimizer = torch.optim.SGD([
        #         {'params': self.projector.parameters()},
        #         {'params': self.encoder.parameters(), 'lr': self.args.lr_base_encoder}
        #     ], lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        #     detach_f = False
        # else:
        #     optimizer = torch.optim.SGD(self.projector.parameters(), lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        #     detach_f = True
        
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_base, weight_decay=self.args.decay)   # wd:1e-6
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optimized_parameters, lr=self.args.lr_base, weight_decay=self.args.decay)  # wd:1e-1
        return optimizer

    def update_fc_ft_novel(self, trainloader, testloader, session):
        theta1 = deepcopy(self.projector.state_dict())
        kp_lam = self.args.kp_lam

        optimizer = self.get_optimizer_new(self.projector.parameters())

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
        # val_freq = 10
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
                    logits = self.fc(encoding)
                    # logits = self.fc.get_dot(encoding)

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

                    loss += KP_loss(theta1, self.projector, lymbda_kp=kp_lam)
                    
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
                    # if epoch % val_freq == 0:
                    #     vl, va, vaNovel, vaBase, vhm, vam, vbin, vaNN, vaBB = self.test_fc_head(self.fc, testloader, epoch, session)
                    #     s = (f"Top 1 Acc: {va*100:.3f}, "
                    #         f"N/j Acc: {vaNovel*100:.3f}, "
                    #         f"B/j Acc: {vaBase*100:.3f}, "
                    #         f"Binary Accuracy: {vbin*100:.3f}")
                    #     print(s)
                        
                tqdm_gen.set_description(out_string)

                scheduler.step()

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
        # vaBase = Averager() # Averager for novel classes only        
        # vaNovel = Averager() # Averager for novel classes only
        vaBinary = Averager() # Averager for binary classification of novel and base classes
        vaNovelN = Averager() # Averager for novel classes only
        vaBaseB = Averager() # Averager for binary classification of novel and base classes

        # test_fc = fc.clone().detach()
        self.eval()

        total_novel_samples = 0
        total_base_samples = 0

        all_probs = []
        all_targets = []

        with torch.no_grad():
            # tqdm_gen = tqdm(testloader)
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]

                encoding = self.encode(data).detach()

                logits = fc.get_logits(encoding)
                logits = logits[:, :test_class]

                all_probs.append(logits)
                all_targets.append(test_label)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                bin_acc = count_acc_binary(logits, test_label, test_class, self.args)

                # >>> Addition
                # novelAcc, baseAcc = count_acc_(logits, test_label, test_class, self.args)

                if session > 0:
                    fpr = count_fp(logits, test_label, test_class, self.args, fpr)
                    novelNAcc, baseBAcc = count_acc_(logits, test_label, test_class, self.args, sub_space="separate")
                    vaNovelN.add(novelNAcc)
                    vaBaseB.add(baseBAcc)

                # vaNovel.add(novelAcc)
                # vaBase.add(baseAcc)
                vaBinary.add(bin_acc)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

            # >>> Addition 
            # vaNovel = vaNovel.item()
            # vaBase = vaBase.item()
            vaBinary = vaBinary.item()
            vaNovelN = vaNovelN.item()
            vaBaseB = vaBaseB.item()

        # Concatenate all_targets and probs
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs, axis=0)

        # Now compute vaNovel as
        novel_mask = all_targets >= self.args.base_class
        pred = torch.argmax(all_probs, dim=1)[novel_mask]
        label_ = all_targets[novel_mask]
        vaNovel = (pred == label_).type(torch.cuda.FloatTensor).mean().item()

        # Compute base acc as
        base_mask = all_targets < self.args.base_class
        pred = torch.argmax(all_probs, dim=1)[base_mask]
        label_ = all_targets[base_mask]
        vaBase = (pred == label_).type(torch.cuda.FloatTensor).mean().item()

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

    def select_criterion(self, base_sess = False):
        criterion = self.args.pull_criterion_base if base_sess else self.args.pull_criterion_novel
        if criterion == "xent":
            return nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        elif criterion == "cosine":
            return torch.nn.CosineEmbeddingLoss()
        elif criterion == "xent+cosine":
            return [nn.CrossEntropyLoss(), torch.nn.CosineEmbeddingLoss()]
        elif criterion == "none":
            return None

    def criterion_forward(self, criterion, logits, label, base_sess = False):
        crit = self.args.pull_criterion_base if base_sess else self.args.pull_criterion_novel
        if crit == "xent":
            return criterion(logits, label)
        elif crit == "cosine":
            target = torch.ones_like(label)
            one_hot_label = F.one_hot(label, logits.shape[1])
            return radial_label_smoothing(criterion, logits, one_hot_label, target, self.args.radial_label_smoothing)
        elif crit == "xent+cosine":
            xent_loss = criterion[0](logits, label)
            target = torch.ones_like(label)
            one_hot_label = F.one_hot(label, logits.shape[1])
            cosine_loss = radial_label_smoothing(criterion[1], logits, one_hot_label, target, self.args.radial_label_smoothing)
            return cosine_loss + self.args.xent_weight*xent_loss
        elif crit == "cosine-squared":
            # https://arxiv.org/pdf/1901.10514.pdf
            # loss = torch.sum(torch.pow(1 - logits[torch.arange(label.shape[0]), label], 2))
            # import pdb; pdb.set_trace()
            loss = (1 - logits[torch.arange(label.shape[0]), label]).pow(2).sum()
            return loss
        elif crit == "none":
            return 0
    
    def update_fc_ft_base(self, baseloader, testloader, class_variance=None):
        if self.args.skip_base_ft:
            return self.test_fc_head(self.fc, testloader, 0, 0)["va"]

        # Optimising only the projector
        if self.args.fine_tune_backbone_base:
            optimized_parameters = [
                {'params': self.projector.parameters()},
                {'params': self.encoder.parameters(), 'lr': self.args.lr_base_encoder}
            ]
            detach_f = False
        else:
            optimized_parameters = self.projector.parameters()
            detach_f = True
            
        optimizer = self.get_optimizer_base(optimized_parameters)

        # Setting up the scheduler
        if self.args.base_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80],gamma=self.args.gamma)
        elif self.args.base_schedule == "Cosine":
            warmup_epochs = self.args.warmup_epochs_base
            min_lr = 1e-5
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, 
                    warmup_epochs=warmup_epochs, 
                    max_epochs=self.args.epochs_base,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_base,
                    eta_min=min_lr)

        sc_criterion = supcon.SupConLoss()
        criterion = self.select_criterion(base_sess = True)

        best_loss = None
        best_hm = None
        best_acc = 0
        best_epoch = 0
        best_projector = None

        # self.eval() # Fixing batch norm

        # target_prototypes class_variance torch.nn.Parameter(self.fc.get_classifier_weights())
        # target_labels = torch.arange(self.args.base_class).cuda()

        # Pulling to targets using all targets
        # target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(), self.fc.rv), axis = 0))
        # target_labels = torch.arange(self.args.num_classes).cuda()

        # Pushing away from novel unassigned targets only
        target_prototypes = torch.nn.Parameter(self.fc.rv)
        target_prototypes.requires_grad = False
        # target_labels = torch.arange(self.args.num_classes - self.args.base_class).cuda() + self.args.base_class
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class

        # Get prototypes and class variances for each class
        # _, prototypeLabels, classVariances, numInstancesPerClass = gaussian_utils.get_prototypes_for_session(None, testloader.dataset.transform, self, self.args, 0, trainloader=baseloader, use_vector_variance = True)
        # average_class_variance = classVariances.mean(axis = 0)  # Now we have an average class variance

        # For simplex loss
        assigned_targets = self.fc.get_classifier_weights().detach().clone()
        assigned_targets_label = torch.arange(self.args.base_class)
        unassigned_targets = self.fc.rv.detach().clone()

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_base))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                for idx, batch in enumerate(baseloader):
                    images, label = batch
                    images = torch.cat([images[0], images[1]], dim=0)
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        label = label.cuda(non_blocking=True)

                    bsz = label.shape[0]

                    projections, encodings = self.encode(images, detach_f=detach_f, return_encodings=True)
                    projections = normalize(projections)

                    # Sup loss with perturbed targets to ensure separation from future classes
                    f1, f2 = torch.split(projections, [bsz, bsz], dim=0)

                    # perturbed_t1 = perturb_targets_w_variance(target_prototypes, average_class_variance)
                    # perturbed_t2 = perturb_targets_w_variance(target_prototypes, average_class_variance)
                    if self.args.skip_perturbation:
                        f1_ = f1
                        f2_ = f2
                        label_ = label
                        pert_count = 0
                    else:
                        if self.args.perturb_dist == "uniform":
                            pbsz = bsz if self.args.batch_size_perturb == -1 else target_prototypes.shape[0]
                            pert, target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), pbsz, 2, epsilon = self.args.perturb_epsilon_base, offset=self.args.perturb_offset)
                            # pert, target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), bsz//2, 2, epsilon = self.args.perturb_epsilon_base)
                        elif self.args.perturb_dist == "gaussian":
                            # Here the epsilon is treated like the std deviation of a normal distribution
                            pert, target_labels_ = perturb_targets_norm_count_gaus(target_prototypes.clone(), target_labels.clone(), bsz, 2, epsilon = self.args.perturb_epsilon_base)
                            
                        perturbed_t1, perturbed_t2 = pert
                        # perturbed_t1 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_base)
                        # perturbed_t2 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_base)
                        # target_labels_ = target_labels.clone()
                        f1_ = torch.cat((f1, perturbed_t1), axis = 0)
                        f2_ = torch.cat((f2, perturbed_t2), axis = 0)
                        label_ = torch.cat((label, target_labels_))
                        pert_count = target_labels.shape[0] if self.args.remove_pert_numerator else 0

                        # Shuffle entire batch
                        # shuff_ix = torch.randperm(f1_.shape[0])
                        # f1_ = f1_[shuff_ix]
                        # f2_ = f2_[shuff_ix]
                        # label_ = label_[shuff_ix]

                    features = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
                    loss = self.args.sup_lam * sc_criterion(features, label_, pert_count = pert_count)

                    # Computing simplex loss
                    sloss = torch.zeros(1)
                    if self.args.simplex_lam > 0:
                        sloss = simplex_loss(f1, label, assigned_targets, assigned_targets_label, unassigned_targets)
                        loss += self.args.simplex_lam * sloss

                    # Add mixco loss
                    if self.args.mix_lam > 0:
                        x0 = images[:bsz, ...]
                        mixup_loss = mix_step(x0, f1, f2, model=self, detach_f=detach_f)
                        loss += self.args.mix_lam * mixup_loss

                    # if self.args.instance_mixup:
                    #     label_rep = label.repeat(2)
                    #     images, label_rep = instance_mixup_data(images, label_rep)
                    #     images, label_rep = map(Variable, (images, label_rep))
                    #     projections, encodings = self.encode(images, detach_f=detach_f, return_encodings=True)
                    #     logits = self.fc(projections)
                    #     loss += self.args.cos_lam * self.criterion_forward(criterion, logits, label_rep, base_sess = True)
                    # else:

                    # Pulling the base classes directly towards the targets.
                    if not self.args.spread_aware:
                        logits = self.fc(projections)
                        label_rep = label.repeat(2)
                        
                        if self.args.apply_tbnce:
                            # compute the logit outputs given the rv for the projections and pass the logits to the criterion forward
                            logits_ext = F.linear(F.normalize(projections, p=2, dim=-1), F.normalize(self.fc.rv, p=2, dim=-1))
                            logits =torch.cat((logits, logits_ext), axis = 1)

                        
                        loss += self.args.cos_lam * self.criterion_forward(criterion, logits, label_rep, base_sess = True)
                    else:
                        # Precompute spread/std dev for each class
                        # Add noise to self.fc befor computing self.fc(projections). Such as to add some noise but also maintain
                        # the spread in the target location

                        logits = self.fc.get_spread_logits(projections, class_variance)
                        label_rep = label.repeat(2)
                        loss += self.args.cos_lam * self.criterion_forward(criterion, logits, label_rep, base_sess = True)
                
                    ta.add(count_acc(logits, label_rep))

                    optimizer.zero_grad()
                    loss.backward()
                    
                    # plot_grad_flow(self.named_parameters())
                    
                    optimizer.step()

                    total_loss += loss.item()

                    out_string = f"Epoch: {epoch}|[{idx}/{len(baseloader)}], Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {best_acc*100:.3f}, simplex_loss: {sloss.item():.3f}"
                    tqdm_gen.set_description(out_string)

                # Model Saving
                test_out = self.test_fc_head(self.fc, testloader, epoch, 0)
                va = test_out["va"]
                
                if best_acc is None or best_acc < va:
                    best_acc = va
                    best_epoch = epoch

                out_string = f"Epoch: {epoch}, Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {va*100:.3f}"
                tqdm_gen.set_description(out_string)
                
                scheduler.step()

        # va = self.test_fc_head(self.fc, testloader, 0, 0)["va"]
        # return va

    def pull_loss(self, label_rep, novel_class_start, criterion, logits, tl, session):
        if self.args.pull_loss_mode == "default":
            novel_classes_idx = torch.argwhere(label_rep >= novel_class_start).flatten()
            base_classes_idx = torch.argwhere(label_rep < novel_class_start).flatten()
            novel_loss = base_loss = 0
            if novel_classes_idx.numel() != 0:
                novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label_rep[novel_classes_idx])                            
                tl["novel"].add(novel_loss.item())
            if base_classes_idx.numel() != 0:
                base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label_rep[base_classes_idx])
                tl["base"].add(base_loss.item())
            cos_loss = (novel_loss*self.args.cos_n_lam) + (base_loss*self.args.cos_b_lam)
        elif self.args.pull_loss_mode == "weighted":
            # For as many sessions as there exist already. Split them and equally weight
            cos_loss = 0
            weight = 1/(session+1)
            for sess in range(session+1):
                if sess == 0:
                    session_idx = torch.argwhere(label_rep < self.args.base_class).flatten()
                else:
                    session_mask = torch.logical_and(label_rep >= self.args.base_class + (self.args.way * (sess - 1)), label_rep < self.args.base_class + (self.args.way*sess)).flatten()
                    session_idx = torch.nonzero(session_mask).squeeze()
                if session_idx.numel() != 0:
                    cos_loss += weight * self.criterion_forward(criterion, logits[session_idx, :], label_rep[session_idx])
        elif self.args.pull_loss_mode == "curr_session":
            # For as many sessions as there exist already. Split them and equally weight
            cos_loss = 0
            weight = 1/(session+1)
            session_idx = torch.argwhere(label_rep > self.args.base_class + (self.args.way * (session - 1))).flatten()
            if session_idx.numel() != 0:
                cos_loss += weight * self.criterion_forward(criterion, logits[session_idx, :], label_rep[session_idx])

        return cos_loss

    def update_fc_ft_simplex_warmup(self, trainloader):
        if self.args.fine_tune_backbone_joint:
            optimizer = torch.optim.SGD([
                {'params': self.projector.parameters()},
                {'params': self.encoder.parameters(), 'lr': self.args.lr_new}
            ], lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        else:
            optimizer = self.get_optimizer_new(self.projector.parameters())
        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.warmup_epochs_simplex))
            for epoch in tqdm_gen:
                for batch in trainloader:
                    images, label = batch
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        label = label.cuda(non_blocking=True)
        
                    projections, _ = self.encode(images, detach_f=True, return_encodings=True)
                    projections = normalize(projections)
                    
                    sloss = simplex_loss_in_batch(projections)

                    optimizer.zero_grad()
                    sloss.backward()
                    optimizer.step()

                tqdm_gen.set_description(f"Simplex loss: {sloss:.3f}")
                    

    def update_fc_ft_joint_supcon(self, jointloader, testloader, class_list, session):
        # Optimising only the projector
        if self.args.fine_tune_backbone_joint:
            optimizer = torch.optim.SGD([
                {'params': self.projector.parameters()},
                {'params': self.encoder.parameters(), 'lr': self.args.lr_new}
            ], lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        else:
            optimizer = self.get_optimizer_new(self.projector.parameters())

        # Setting up the scheduler
        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120],gamma=self.args.gamma)
        else:
            warmup_epochs = self.args.warmup_epochs_inc
            # min_lr = self.args.lr_new / 2. # orignally 0
            min_lr = 0
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, 
                    warmup_epochs=warmup_epochs, 
                    max_epochs=self.args.epochs_joint,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                    eta_min=min_lr)

        sc_criterion = supcon.SupConLoss()
        pull_criterion = self.select_criterion()
        xent_criterion = nn.CrossEntropyLoss()

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
        elif self.args.queue_pull:
            queued_session = math.floor(session / 2)
            novel_class_start = self.args.base_class + (self.args.way * (queued_session))
        else:
            novel_class_start = self.args.base_class

        self.eval() # Turn on

        test_freq = self.args.testing_freq
        
        # target_prototypes = torch.nn.Parameter(self.fc.get_classifier_weights())
        # target_labels = torch.arange(self.args.base_class + (self.args.way * session)).cuda()

        # target_prototypes = torch.nn.Parameter(self.fc.rv)
        # target_labels = torch.arange(self.args.num_classes - (self.args.base_class + (self.args.way * session)) ).cuda() + (self.args.base_class + (self.args.way * session))

        if self.args.perturb_mode == "inc-curr-base": #"all", "inc+curr-base", "inc-curr+base"
            # Target prototypes contains assigned targets and unassigned targets (except base and current session targets)
            target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session)[self.args.base_class:].clone(), self.fc.rv.clone())))
            target_prototypes.requires_grad = False
            # target_labels = torch.arange(self.args.num_classes - self.args.base_class).cuda() + self.args.base_class
            target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class
            curr_labels = torch.arange(self.args.way * (session - 1), self.args.way * session).cuda() + self.args.base_class
            mask = ~torch.isin(target_labels, curr_labels)
            target_labels = target_labels[mask]
        elif self.args.perturb_mode == "inc+curr-base":
            # Target prototypes contains assigned targets and unassigned targets (except base)
            target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session+1)[self.args.base_class:].clone(), self.fc.rv.clone())))
            target_prototypes.requires_grad = False
            target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class
        elif self.args.perturb_mode == "inc-curr+base":
            # Target prototypes contains assigned targets and unassigned targets (except base)
            target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session).clone(), self.fc.rv.clone())))
            target_prototypes.requires_grad = False
            target_labels = torch.arange(self.fc.reserve_vector_count).cuda()
            curr_labels = torch.arange(self.args.way * (session - 1), self.args.way * session).cuda() + self.args.base_class
            mask = ~torch.isin(target_labels, curr_labels)
            target_labels = target_labels[mask] # Removing curr labels from our target labels
        elif self.args.perturb_mode == "all":
            target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session+1).clone(), self.fc.rv.clone())))
            target_prototypes.requires_grad = False
            target_labels = torch.arange(self.fc.reserve_vector_count).cuda()
        elif self.args.perturb_mode == "all-ua":
            target_prototypes = torch.nn.Parameter((self.fc.get_classifier_weights(uptil=session+1)))
            target_prototypes.requires_grad = False
            target_labels = torch.arange(self.args.base_class + (session*self.args.way)).cuda()
        elif self.args.perturb_mode == "inc+curr-base-ua":
            # Target prototypes contains assigned targets and unassigned targets (except base)
            target_prototypes = torch.nn.Parameter((self.fc.get_classifier_weights(uptil=session+1)[self.args.base_class:].clone()))
            target_prototypes.requires_grad = False
            target_labels = torch.arange(session * self.args.way).cuda() + self.args.base_class


        # For simplex loss
        assigned_targets = self.fc.get_classifier_weights().detach().clone()
        assigned_targets_label = torch.arange(self.args.base_class + (self.args.way * session))
        unassigned_targets = self.fc.rv.detach().clone()

        out_string = ""

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                sc_total_loss = 0
                pull_total_loss = 0
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

                    projections, encodings = self.encode(images, detach_f=True, return_encodings=True)

                    # Normalise projections
                    projections = normalize(projections)

                    f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
                    

                    # Add more labels for newer sessions
                    # new_ixs = torch.argwhere(label >= (self.args.base_class + (self.args.way * (session-1)))).flatten()
                    # old_ixs = torch.argwhere(label < (self.args.base_class + (self.args.way * (session-1)))).flatten()
                    new_ixs = torch.argwhere(label >= self.args.base_class).flatten()
                    old_ixs = torch.argwhere(label < self.args.base_class).flatten()

                    # Perturbation scheudling
                    # f1_ = torch.cat((f1, perturb_targets_norm(target_prototypes[label[old_ixs]], epsilon=1e-1), perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=1e-1*(0.99**epoch))), axis = 0)
                    # f2_ = torch.cat((f2, perturb_targets_norm(target_prototypes[label[old_ixs]], epsilon=5e-1), perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=1e-1*(0.99**epoch))), axis = 0)

                    # Perturbation using interpolation
                    # f1_ = torch.cat((f1, perturb_targets_norm(target_prototypes[label[old_ixs]], epsilon=1e-1), interpolate_target(target_prototypes[label[new_ixs]], f1[new_ixs]) ), axis = 0)
                    # f2_ = torch.cat((f2, perturb_targets_norm(target_prototypes[label[old_ixs]], epsilon=5e-1), interpolate_target(target_prototypes[label[new_ixs]], f2[new_ixs]) ), axis = 0)
                    # label_ = torch.cat((label, label[old_ixs], label[new_ixs]))

                    # Phased targetting
                    # if epoch < 50:
                    #     f1_ = torch.cat((f1, perturb_targets_norm(target_prototypes[label[old_ixs]], epsilon=1e-2), perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=1e-2)), axis = 0)
                    #     f2_ = torch.cat((f2, perturb_targets_norm(target_prototypes[label[old_ixs]], epsilon=5e-2), perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=5e-2)), axis = 0)
                    #     label_ = torch.cat((label, label[old_ixs], label[new_ixs]))
                    # else:
                    #     # new_ixs = torch.argwhere(label >= (self.args.base_class + (self.args.way * (session-1)))).flatten()
                    #     # inter_ix = torch.argwhere(torch.logical_and(label >= (self.args.base_class + (self.args.way * (session-1))), label >= self.args.base_class)).flatten()
                    #     # f1_ = torch.cat((f1, perturb_targets_norm(target_prototypes[label[inter_ix]], epsilon=1e-1), perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=1e-2)), axis = 0)
                    #     # f2_ = torch.cat((f2, perturb_targets_norm(target_prototypes[label[inter_ix]], epsilon=5e-1), perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=5e-2)), axis = 0)
                    #     # label_ = torch.cat((label, label[inter_ix], label[new_ixs]))
                    #     f1_ = torch.cat((f1, perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=1e-2)), axis = 0)
                    #     f2_ = torch.cat((f2, perturb_targets_norm(target_prototypes[label[new_ixs]], epsilon=5e-2)), axis = 0)
                    #     label_ = torch.cat((label, label[new_ixs]))

                    # Sup loss with perturbed targets for base classes
                    # old_ixs = torch.argwhere(label < (self.args.base_class + (self.args.way * (session-1)))).flatten()
                    # f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
                    # old_ixs = torch.argwhere(target_labels < self.args.base_class).flatten()
                    # f1_ = torch.cat((f1, perturb_targets_norm(target_prototypes[old_ixs], epsilon=1e-2)), axis = 0)
                    # f2_ = torch.cat((f2, perturb_targets_norm(target_prototypes[old_ixs], epsilon=1e-2)), axis = 0)
                    # label_ = torch.cat((label, target_labels[old_ixs]))

                    loss = 0    

                    # Perturbing only unassigned and previous session targets and using cosine loss to pull novel classes together
                    if self.args.skip_perturbation:
                        f1_ = f1
                        f2_ = f2
                        label_ = label
                        pert_count = 0
                        pert_count = 0
                    else:
                        # import pdb; pdb.set_trace()
                        if self.args.remove_curr_features:
                            mask = ~torch.isin(label, curr_labels)
                            label_ = label[mask]
                            f1_ = f1[mask]
                            f2_ = f2[mask]
                        else:
                            f1_ = f1
                            f2_ = f2
                            label_ = label

                        if self.args.perturb_dist == "uniform":
                            pbsz = bsz if self.args.batch_size_perturb == -1 else target_prototypes.shape[0]
                            pert, target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), pbsz, 2, epsilon = self.args.perturb_epsilon_inc, offset=self.args.perturb_offset)
                            # pert, target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), bsz//2, 2, epsilon = self.args.perturb_epsilon_inc)
                        elif self.args.perturb_dist == "gaussian":
                            pert, target_labels_ = perturb_targets_norm_count_gaus(target_prototypes.clone(), target_labels.clone(), bsz, 2, epsilon = self.args.perturb_epsilon_inc)

                        perturbed_t1, perturbed_t2 = pert
                        # perturbed_t1 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_inc)
                        # perturbed_t2 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_inc)
                        # target_labels_ = target_labels.clone()
                        f1_ = torch.cat((f1_, perturbed_t1), axis = 0)
                        f2_ = torch.cat((f2_, perturbed_t2), axis = 0)
                        label_ = torch.cat((label_, target_labels_))
                        pert_count = target_labels.shape[0] if self.args.remove_pert_numerator else 0
                        # pert_count = 0
                        
                        # Shuffle entire batch
                        # shuff_ix = torch.randperm(f1_.shape[0])
                        # f1_ = f1_[shuff_ix]
                        # f2_ = f2_[shuff_ix]
                        # label_ = label_[shuff_ix]
                        
                    features = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
                    sc_loss = self.args.sup_lam * sc_criterion(features, label_, pert_count = pert_count)
                    loss += sc_loss

                    # Computing simplex loss
                    if self.args.simplex_lam_inc > 0:
                        sloss = simplex_loss(f1[new_ixs], label, assigned_targets, assigned_targets_label, unassigned_targets)
                        # sloss = simplex_loss(f1, label, assigned_targets, assigned_targets_label, unassigned_targets)
                        loss += self.args.simplex_lam_inc * sloss

                    # Add mixco loss
                    # if self.args.mix_lam > 0:
                    #     x0 = images[:bsz, ...]
                    #     mixup_loss = mix_step(x0, f1, f2)
                    #     loss += self.args.mix_lam * mixup_loss

                    if self.args.novel_bias_schedule != -1 and epoch > self.args.novel_bias_schedule and self.args.novel_bias:
                        # Schedule the novel bias
                        novel_class_start = self.args.base_class                        

                    # Cosine Pull Loss
                    logits = self.fc(projections)
                    label_rep = label.repeat(2)
                    # if self.args.apply_tbnce:
                    #     # compute the logit outputs given the rv for the projections and pass the logits to the criterion forward
                    #     logits_ext = F.linear(F.normalize(projections, p=2, dim=-1), F.normalize(self.fc.rv, p=2, dim=-1))
                    #     logits =torch.cat((logits, logits_ext), axis = 1)
                    pull_loss = self.args.cos_lam * self.pull_loss(label_rep, novel_class_start, pull_criterion, logits, tl, session)
                    loss += pull_loss

                    # xent pull
                    # old_ixs = torch.argwhere(target_labels < (self.args.base_class + (self.args.way * (session-1)))).flatten()
                    # new_ixs = torch.argwhere(label_rep >= (self.args.base_class + (self.args.way * (session-1)))).flatten()
                    # Concat only the novel classes with perturbed targets for previous classes and try to pull and push everything in position
                    # projections_ = torch.cat((projections, perturb_targets_norm(target_prototypes[old_ixs], epsilon=1e-2)), axis = 0)
                    # labels_ = torch.cat((label_rep, target_labels[old_ixs]))
                    # logits_ = self.fc(projections)
                    # pull_loss = xent_criterion(logits[new_ixs], label_rep[new_ixs])

                    ta.add(count_acc(logits, label_rep))

                    # Update projector ema
                    if self.args.proj_ema_update:
                        # When applying to novel classes it requires that the logits remain but that cannot be true
                        if self.args.proj_ema_mode == "prev":
                            base_classes_idx = torch.argwhere(label < (self.args.base_class + (self.args.way * (session-1)))).flatten()
                        elif self.args.proj_ema_mode == "all":
                            base_classes_idx = torch.arange(label.shape[0])
                        elif self.args.proj_ema_mode == "base":
                            base_classes_idx = torch.argwhere(label < self.args.base_class).flatten()
                        if base_classes_idx.numel() != 0:
                            e1, _ = torch.split(encodings, [bsz, bsz], dim=0)

                            teacher_logits = self.projector_ema.ema_model(e1[base_classes_idx]).detach()       # Untrack gradients on this path
                            teacher_logits = normalize(teacher_logits)

                            # Proj ema without softmax
                            teacher_logits = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1)
                            # teacher_logits = F.softmax(teacher_logits / self.teacher_temp, dim=-1)
                            
                            student_logits = F.softmax(f1[base_classes_idx] / self.student_temp, dim=-1)
                            # student_logits = F.softmax(f2[base_classes_idx] / self.student_temp, dim=-1)

                            # Update teacher center
                            self.update_center(teacher_logits)

                            # Now apply cross entropy
                            dist_loss = F.cross_entropy(teacher_logits, student_logits)
                            loss += self.args.dist_lam * dist_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if self.args.proj_ema_update:
                        self.projector_ema.update()

                    sc_total_loss += sc_loss.item()
                    pull_total_loss += pull_loss.item()

                # Model Saving
                # if (epoch + 1) % test_freq == 0:
                if epoch % test_freq == 0:    # original
                    if self.args.validation_metric == "hm":
                        metrics = self.test_fc_head(self.fc, testloader, epoch, session)
                        # Validation
                        vhm = metrics["vhm"]
                        if best_hm is None or vhm > best_hm:
                            best_hm = vhm
                            best_hm_epoch = epoch
                            hm_patience_count = -1
                            best_projector = deepcopy(self.projector.state_dict())
                            if self.args.fine_tune_backbone_joint:
                                best_l4 = deepcopy(self.encoder.layer4.state_dict())
                        out_string = '(Joint) LR:{:.3f}, Sess: {}, loss: sc/pull {:.3f}/{:.3f}|(b/n)({:.3f}/{:.3f}), trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (test) b/n={:.3f}/{:.3f}, bb/nn={:.3f}/{:.3f}'\
                            .format(
                                    scheduler.get_last_lr()[0],
                                    session, 
                                    sc_total_loss,
                                    pull_total_loss,
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
                        best_projector = self.projector.state_dict()
                        if self.args.fine_tune_backbone_joint:
                            best_l4 = self.encoder.layer4.state_dict()
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
        
        self.projector.load_state_dict(best_projector, strict=True)
        if self.args.fine_tune_backbone_joint:
            self.encoder.layer4.load_state_dict(best_l4, strict = True)        

        # if self.args.proj_ema_update:
        #     self.projector.load_state_dict(self.projector_ema.ema_model.state_dict(), strict=True)
        
        
    def visualise_grad_flow(self, base_loader, inc_loader):        
        # Run the base loader for several batches without optimizer.step

        # Compute the sup con loss, xent loss and simplex loss
        
        # loss.backward
        
        # Skip optimizer.step to avoid gradient update

        # Plot the visualisation as "grad_flow_base_{session}.png"

        # Plot the visualisation as "grad_flow_base_{session}.png"
        return