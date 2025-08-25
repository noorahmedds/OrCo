import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR
import supcon

from torchvision.models import resnet18 as tv_resnet18
from models.resnet18 import ResNet18
from models.resnet12 import resnet12_nc

from helper import *
from utils import *
from copy import deepcopy

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

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

    def get_assignment(self, cost):
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
        col_ind = self.get_assignment(cost)
        new_fc_tensor = self.rv[col_ind]

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
        col_ind = self.get_assignment(cost)
        new_fc_tensor = self.rv[col_ind]

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
        tqdm_gen = tqdm(range(self.args.epochs_target_gen))

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
            self.encoder = ResNet18()  
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

    def update_targets(self, trainloader, testloader, class_list, session):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        new_prototypes, _ = self.get_class_avg(data, label, class_list)

        # Assign a new novel classifier from the given reseve vectors
        self.fc.assign_novel_classifier(new_prototypes)

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer_joint == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        return optimizer

    def get_optimizer_base(self, optimized_parameters):        
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        return optimizer

    def test_pseudo_targets(self, fc, testloader, epoch, session):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        va = Averager()

        self.eval()
        with torch.no_grad():
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]
                encoding = self.encode(data).detach()
                logits = fc.get_logits(encoding)
                logits = logits[:, :test_class]
                acc = count_acc(logits, test_label)
                va.add(acc)
            va = va.item()

        metrics = {
            "va": va
        }

        return metrics
    
    def select_criterion(self):
        return nn.CrossEntropyLoss()

    def criterion_forward(self, criterion, logits, label):
        return criterion(logits, label)
    
    def pull_loss(self, label_rep, novel_class_start, criterion, logits):
        novel_classes_idx = torch.argwhere(label_rep >= novel_class_start).flatten()
        base_classes_idx = torch.argwhere(label_rep < novel_class_start).flatten()
        novel_loss = base_loss = 0
        if novel_classes_idx.numel() != 0:
            novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label_rep[novel_classes_idx])                            
        if base_classes_idx.numel() != 0:
            base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label_rep[base_classes_idx])
        cos_loss = (novel_loss*self.args.cos_n_lam) + (base_loss*self.args.cos_b_lam)
        return cos_loss

    def update_base(self, baseloader, testloader):
        if self.args.fine_tune_backbone_base:
            # For CUB it is critical to also train the backbone
            optimized_parameters = [
                {'params': self.projector.parameters()},
                {'params': self.encoder.parameters(), 'lr': self.args.lr_base_encoder}
            ]
            detach_f = False
        else:
            # Optimising only the projector
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
            
        scl = supcon.SupConLoss()
        xent = self.select_criterion()

        best_acc = 0
        best_projector = None

        # Targets for PSCL
        target_prototypes = torch.nn.Parameter(self.fc.rv)
        target_prototypes.requires_grad = False
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class

        # Variables for Orthogonality Loss
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

                    bsz = pbsz = label.shape[0]

                    projections, _ = self.encode(images, detach_f=detach_f, return_encodings=True)
                    projections = normalize(projections)

                    # PSCL Loss
                    f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
                    (perturbed_t1, perturbed_t2), target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), pbsz, 2, epsilon = self.args.perturb_epsilon_base, offset=self.args.perturb_offset)
                    f1_ = torch.cat((f1, perturbed_t1), axis = 0)
                    f2_ = torch.cat((f2, perturbed_t2), axis = 0)
                    label_ = torch.cat((label, target_labels_))
                    features = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
                    loss = self.args.sup_lam * scl(features, label_)

                    # Orthogonality Loss
                    orth_loss = simplex_loss(f1, label, assigned_targets, assigned_targets_label, unassigned_targets)
                    loss += self.args.simplex_lam * orth_loss

                    # Cross Entropy
                    logits = self.fc(projections)
                    label_rep = label.repeat(2)
                    loss += self.args.cos_lam * self.criterion_forward(xent, logits, label_rep)
                
                    ta.add(count_acc(logits, label_rep))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    out_string = f"Epoch: {epoch}|[{idx}/{len(baseloader)}], Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {best_acc*100:.3f}"
                    tqdm_gen.set_description(out_string)

                # Model Saving
                test_out = self.test_pseudo_targets(self.fc, testloader, epoch, 0)
                va = test_out["va"]
                
                if best_acc is None or best_acc < va:
                    best_acc = va
                    best_projector = deepcopy(self.projector.state_dict())

                out_string = f"Epoch: {epoch}, Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {va*100:.3f}"
                tqdm_gen.set_description(out_string)
                
                scheduler.step()

        # Setting to best validation projection head
        self.projector.load_state_dict(best_projector, strict=True)

    def update_incremental(self, jointloader, session):
        # Optimising only the projector
        optimizer = self.get_optimizer_new(self.projector.parameters())

        # Setting up the Cosine Scheduler
        warmup_epochs = self.args.warmup_epochs_inc
        min_lr = 0
        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, 
                warmup_epochs=warmup_epochs, 
                max_epochs=self.args.epochs_joint,
                warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                eta_min=min_lr)

        sc_criterion = supcon.SupConLoss()
        pull_criterion = self.select_criterion()

        best_projector = None
        novel_class_start = self.args.base_class

        self.eval()

        # Target prototypes contains assigned targets and unassigned targets (except base)
        target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session+1)[self.args.base_class:].clone(), self.fc.rv.clone())))
        target_prototypes.requires_grad = False
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class

        # For Orthogonality loss
        assigned_targets = self.fc.get_classifier_weights().detach().clone()
        assigned_targets_label = torch.arange(self.args.base_class + (self.args.way * session))
        unassigned_targets = self.fc.rv.detach().clone()

        out_string = ""

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                for idx, batch in enumerate(jointloader):
                    images, label = batch
                    images = torch.cat([images[0], images[1]], dim=0)
                    if torch.cuda.is_available():
                        images = images.cuda(non_blocking=True)
                        label = label.cuda(non_blocking=True)

                    bsz = label.shape[0]

                    projections, _ = self.encode(images, detach_f=True, return_encodings=True)
                    projections = normalize(projections)
                    f1, f2 = torch.split(projections, [bsz, bsz], dim=0)
                    
                    # Add more labels for newer sessions
                    new_ixs = torch.argwhere(label >= self.args.base_class).flatten()

                    # Perturbing only unassigned and previous session targets and using cosine loss to pull novel classes together
                    pbsz = bsz
                    (perturbed_t1, perturbed_t2), target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), pbsz, 2, epsilon = self.args.perturb_epsilon_inc, offset=self.args.perturb_offset)
                    f1_ = torch.cat((f1, perturbed_t1), axis = 0)
                    f2_ = torch.cat((f2, perturbed_t2), axis = 0)
                    label_ = torch.cat((label, target_labels_))
                    features = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
                    pscl = self.args.sup_lam * sc_criterion(features, label_)

                    # Computing simplex loss
                    orth_loss = self.args.simplex_lam * simplex_loss(f1[new_ixs], label[new_ixs], assigned_targets, assigned_targets_label, unassigned_targets)             

                    # Cross Entropy
                    logits = self.fc(projections)
                    label_rep = label.repeat(2)
                    xent_loss = self.args.cos_lam * self.pull_loss(label_rep, novel_class_start, pull_criterion, logits)

                    # Combined Loss
                    loss = pscl + xent_loss + orth_loss

                    ta.add(count_acc(logits, label_rep))
                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Model Saving
                out_string = 'Session: {}, Epoch: {}|, Training Loss (Joint): {:.3f}, Training Accuracy (Joint): {:.3f}'\
                    .format(
                            session, 
                            epoch,
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0))
                            )
                tqdm_gen.set_description(out_string)
                
                scheduler.step()