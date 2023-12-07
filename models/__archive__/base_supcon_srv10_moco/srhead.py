import numpy as np
import torch.nn as nn
import torch
from helper import *
from utils import *

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm

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
            elif self.args.reserve_mode in ["full"]:
                self.reserve_vector_count = self.num_features

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
        if self.args.base_target_sampling:
            target_choice_ix = self.args.base_class

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        elif self.args.assign_similarity_metric == "mahalanobis":
            cost = compute_pairwise_mahalanobis(base_prototypes.cpu(), base_cov, self.rv.cpu()[:target_choice_ix])

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

        if self.args.assign_similarity_metric == "cos":
            cost = cosine_similarity(new_prototypes.cpu(), self.rv.cpu())
        elif self.args.assign_similarity_metric == "euclidean":
            cost = euclidean_distances(new_prototypes.cpu(), self.rv.cpu())
        elif self.args.assign_similarity_metric == "mahalanobis":
            cost = compute_pairwise_mahalanobis(new_prototypes.cpu(), cov_list, self.rv.cpu())

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
        # return self.get_logits(x)
        return self.get_dot(x)
    
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