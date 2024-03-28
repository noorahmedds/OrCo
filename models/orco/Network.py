import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR
import supcon

from torchvision.models import resnet18 as tv_resnet18
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12_encoder import *
from models.alice_model import resnet_CIFAR
from models.resnet12_nc import *

from helper import *
from utils import *
from copy import deepcopy

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

# from ema_pytorch import EMA

import math

class PseudoTargetClassifier(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        
        self.args = args
        self.num_features = num_features        # Input dimension for all classifiers

        # Classifier for the base classes
        self.base_fc = nn.Linear(self.num_features, self.args.base_class, bias=False)       # Note the entire number of classes are already added

        # Set of all classifiers
        self.classifiers = nn.Sequential(self.base_fc)

        # Register buffer for the pseudo targets. Assume the total number of classes
        self.num_classes = self.args.num_classes
        self.n_inc_classes = self.args.num_classes - self.args.base_class

        # Number of generated pseudo targets
        if self.args.reserve_mode in ["all"]:
            self.reserve_vector_count = self.num_classes
        elif self.args.reserve_mode in ["full"]:
            self.reserve_vector_count = self.num_features

        # Storing the generated pseudo targets (reserved vectors)
        self.register_buffer("rv", torch.randn(self.reserve_vector_count, self.num_features))

        self.temperature = 1.0

    def compute_angles(self, vectors):
        proto = vectors.cpu().numpy()
        dot = np.matmul(proto, proto.T)
        dot = dot.clip(min=0, max=1)
        theta = np.arccos(dot)
        np.fill_diagonal(theta, np.nan)
        theta = theta[~np.isnan(theta)].reshape(theta.shape[0], theta.shape[1] - 1)
        
        avg_angle_close = theta.min(axis = 1).mean()
        avg_angle = theta.mean()

        return np.rad2deg(avg_angle), np.rad2deg(avg_angle_close)

    def get_assignment(self, cost, assignment_mode):
        """Tak array with cosine scores and return the output col ind """
        _, col_ind = linear_sum_assignment(cost, maximize = True)
        return col_ind

    def get_classifier_weights(self, uptil = -1):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            if uptil >= 0 and uptil < i + 1:
                break
            output.append(cls.weight.data)
        return torch.cat(output, axis = 0)

    def assign_base_classifier(self, base_prototypes):
        # Normalise incoming prototypes
        base_prototypes = normalize(base_prototypes)

        target_choice_ix = self.reserve_vector_count

        cost = cosine_similarity(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
 
        col_ind = self.get_assignment(cost, self.args.assignment_mode_base)
        
        new_fc_tensor = self.rv[col_ind]

        avg_angle, avg_angle_close = self.compute_angles(new_fc_tensor)
        print(f"Selected Base Classifiers have average angle: {avg_angle} and average closest angle: {avg_angle_close}")

        # Create fixed linear layer
        self.classifiers[0].weight.data = new_fc_tensor

        # Remove from the final rv
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]

    def assign_novel_classifier(self, new_prototypes):  
        # Normalise incoming prototypes
        new_prototypes = normalize(new_prototypes)

        target_choice_ix = self.reserve_vector_count

        cost = cosine_similarity(new_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
            
        col_ind = self.get_assignment(cost, self.args.assignment_mode_novel, new_prototypes)
        
        new_fc_tensor = self.rv[col_ind]

        avg_angle, avg_angle_close = self.compute_angles(new_fc_tensor)
        print(f"Selected Novel Classifiers have average angle: {avg_angle} and average closest angle: {avg_angle_close}")

        # Creating and appending a new classifier from the given reserved vectors
        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0], bias=False).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        self.classifiers.append(new_fc.cuda())

        # Maintaining the pseudo targets. Self.rv contains only the unassigned vectors
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]


    def find_reseverve_vectors_all(self):
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
            
            l.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = self.compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data

    def forward(self, x):
        return self.get_logits(x)
        
    def get_logits(self, encoding, session = 0):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            out = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            out = out / self.temperature
            output.append(out)
        output = torch.cat(output, axis = 1)
        return output

        
class ORCONET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset == 'cifar100':
            self.encoder = resnet12_nc()
            self.encoder_outdim = 640
            self.proj_hidden_dim = 2048
            self.proj_output_dim = 128
        if self.args.dataset == 'mini_imagenet':
            self.encoder = resnet_CIFAR.ResNet18()  
            self.encoder.fc = nn.Identity()         # Bypassing the fully connected layer    
            self.encoder_outdim = 512
            self.proj_hidden_dim = 2048
            self.proj_output_dim = 128
        if self.args.dataset == 'cub200':
            self.encoder = tv_resnet18(pretrained=True)
            self.encoder.fc = nn.Identity()         # Bypassing the fully connected layer
            self.encoder_outdim = 512
            self.proj_hidden_dim = 2048
            self.proj_output_dim = 256

        # Select the projection head ('g' from the main paper)
        self.projector = self.select_projector()
        self.best_projector = None  # Stores the best projection head after phase 1

        # Final classifier. This hosts the pseudo targets, all and classification happens here
        self.fc = PseudoTargetClassifier(self.args, self.proj_output_dim)

    def set_projector(self):
        self.best_projector = deepcopy(self.projector.state_dict())

    def reset_projector(self):
        self.projector.load_state_dict(self.best_projector)

    def select_projector(self):
        if self.args.proj_type == "proj":
            # projector
            projector = nn.Sequential(
                nn.Linear(self.encoder_outdim, self.proj_hidden_dim),
                nn.ReLU(),
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

    def update_fc(self, trainloader, testloader, class_list, session, mode="encoder"):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            if mode=="encoder":
                data=self.encode(data).detach()
            elif mode == "backbone":
                data=self.encoder(data).detach()

        new_prototypes, _ = self.get_class_avg(data, label, class_list)

        # Assign a new novel classifier from the given reseve vectors
        if mode == "encoder":
            self.fc.assign_novel_classifier(new_prototypes)

    # def get_logits(self,x,fc):
    #     return F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)) # self.args.temperature

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer_joint == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9 if not self.args.nesterov_new else 0, weight_decay=self.args.decay_new, nesterov=self.args.nesterov_new)
        return optimizer

    def get_optimizer_base(self, optimized_parameters):        
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        return optimizer

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
    
    def select_criterion(self):
        return nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

    def criterion_forward(self, criterion, logits, label):
        return criterion(logits, label)
    
    def update_fc_ft_base(self, baseloader, testloader, class_variance=None):
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
                                    sc_total_loss + pull_total_loss,
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
