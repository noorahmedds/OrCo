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
from srhead import *
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

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
        
class MYNET(nn.Module):

    def __init__(self, args, mode=None, writer=None, K=512, m=0.999, T=0.07):
        super().__init__()

        self.mode = mode
        self.args = args
        
        self.K = 512  # queue_size 65536
        self.m = m  # MOCO momentum
        self.T = T  # Softmax temperature
        
        if self.args.dataset in ['cifar100']:
            # == Resnet12_nc
            self.encoder = resnet12_nc()
            self.encoder_k = resnet12_nc()
            
            self.encoder_outdim = 640
            self.proj_hidden_dim = 2048
            self.proj_output_dim = 128 # 128

        self.writer = writer

        # Sup con projection also the projector which gets fine tuned during the joint session
        self.projector = self.select_projector()
        self.projector_k = self.select_projector()

        self.predictor = self.select_projector()

        # FC is created from the mean outputs of the final linear layer of the projector for all class samples
        self.fc = SRHead(self.args, self.proj_output_dim)
        
        self.register_buffer("queue", torch.randn(self.proj_output_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("label_queue", torch.zeros(self.K).long())
        self.label_queue -= 1

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_key_encoder(self):
        # Copying parameter data between both encoders
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(
            self.projector.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.projector.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def select_projector(self):
        projector = nn.Sequential(
            nn.Linear(self.encoder_outdim, self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )
        return projector

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
    
    def encode_k(self, x, detach_f = False, return_encodings = False):
        encodings = self.encoder_k(x)

        # Only finetuning the projector
        if detach_f: encodings = encodings.detach()

        x = self.projector_k(encodings)

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
        
        im_q = input
        im_k = kwargs["im_k"]
        labels = kwargs["labels"]
        
        if "perturbed_t1" in kwargs:
            pert_t1 = kwargs["perturbed_t1"]
            pert_t2 = kwargs["perturbed_t2"]
            pert_labels = kwargs["pert_labels"]
        
        # ===== MOCO forward
        q, q_enc = self.encode(im_q, return_encodings=True)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encode_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        # Append perturbations to k and to the labels
        if "perturbed_t1" in kwargs:
            q_ = torch.cat((q, pert_t1), axis = 0)
            k = torch.cat((k, pert_t2), axis = 0)
            labels = torch.cat((labels, pert_labels))
        
        keys_curbatch = concat_all_gather(k)  # BxD
        labels_curbatch = concat_all_gather(labels) # B
        queue_ = self.queue.clone().detach()  # DxK
        labels_ = self.label_queue.clone().detach()
        
        queue_enqueue = torch.cat([keys_curbatch.T, queue_], dim=1)  # Dx(B+K)
        labels_enqueue = torch.cat([labels_curbatch, labels_], dim=0)  # B+K
        
        mask = torch.eq(labels[:, None], labels_enqueue[:, None].T)  # bx(B+K)
        
        logits = torch.einsum('nc,ck->nk', [q_, queue_enqueue.clone().detach()]).div(self.T)
        
        loss_sup = (-torch.log_softmax(logits, dim=1) * mask).sum(dim=-1, keepdim=True).div(mask.sum(dim=-1, keepdim=True) + 1e-5)
        loss = loss_sup.mean()        

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        return q, k, loss, q_enc

    def get_logits(self,x,fc):
        return F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)) # self.args.temperature

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

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer_joint == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9 if not self.args.nesterov_new else 0, weight_decay=self.args.decay_new, nesterov=self.args.nesterov_new)
        elif self.args.optimizer_joint == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)   # wd:1e-6
        elif self.args.optimizer_joint == "adamw":
            optimizer = torch.optim.AdamW(optimized_parameters, lr=self.args.lr_new, weight_decay=self.args.decay_new)  # wd:1e-1
        return optimizer

    def get_optimizer_base(self, optimized_parameters):
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_base, weight_decay=self.args.decay)   # wd:1e-6
        elif self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optimized_parameters, lr=self.args.lr_base, weight_decay=self.args.decay)  # wd:1e-1
        return optimizer

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
            self.args.cos_b_lam = 0 if self.args.cos_b_lam > 0 else 0.5
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
    
    def update_fc_ft_base(self, baseloader, testloader, class_variance=None):
        if self.args.skip_base_ft:
            return self.test_fc_head(self.fc, testloader, 0, 0)["va"]

        # Optimising only the projector
        if self.args.fine_tune_backbone_base:
            optimized_parameters = [
                {'params': self.projector.parameters()},
                {"params": self.predictor.parameters()},
                {'params': self.encoder.parameters(), 'lr': self.args.lr_base_encoder}
            ]
            detach_f = False
        else:
            optimized_parameters = [
                {"params": self.projector.parameters()},
                {"params": self.predictor.parameters()}
                ]
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

        # Pushing away from novel unassigned targets only
        target_prototypes = torch.nn.Parameter(self.fc.rv)
        target_prototypes.requires_grad = False
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class

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
                    # images = torch.cat([images[0], images[1]], dim=0)
                    if torch.cuda.is_available():
                        images = [i.cuda(non_blocking=True) for i in images]
                        label = label.cuda(non_blocking=True)

                    bsz = label.shape[0]
                    
                    # pert_count = int(math.pow(2, math.ceil(math.log(target_labels.shape[0] + bsz)/math.log(2))) - bsz)
                    pert_count = bsz
                    pert, target_labels = perturb_targets_norm_count(target_prototypes, target_labels, pert_count, 2, epsilon = self.args.perturb_epsilon_base)
                    perturbed_t1, perturbed_t2 = pert
                    projections, _, sc_loss, encodings = self.forward(images[0], im_k = images[1], labels=label, 
                                                           perturbed_t1 = perturbed_t1, perturbed_t2 = perturbed_t2, pert_labels = target_labels)

                    # if self.args.skip_perturbation:
                    #     f1_ = f1
                    #     f2_ = f2
                    #     label_ = label
                    # else:
                    #     perturbed_t1 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_base)
                    #     perturbed_t2 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_base)
                    #     f1_ = torch.cat((f1, perturbed_t1), axis = 0)
                    #     f2_ = torch.cat((f2, perturbed_t2), axis = 0)
                    #     label_ = torch.cat((label, target_labels))
                    #     pert_count = target_labels.shape[0] if self.args.remove_pert_numerator else 0
                    # features = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
                    # loss = self.args.sup_lam * sc_criterion(features, label_, self.queue_labels, pert_count = pert_count)
                    loss = self.args.sup_lam * sc_loss

                    # Computing simplex loss
                    sloss = torch.zeros(1)
                    if self.args.simplex_lam > 0:
                        sloss = simplex_loss(projections, label, assigned_targets, assigned_targets_label, unassigned_targets)
                        loss += self.args.simplex_lam * sloss

                    # Pulling the base classes directly towards the targets.
                    logits = self.fc(self.predictor(encodings))
                    loss += self.args.cos_lam * self.criterion_forward(criterion, logits, label, base_sess = True)

                    ta.add(count_acc(logits, label))

                    optimizer.zero_grad()
                    loss.backward()
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

    # def update_fc_ft_joint_supcon(self, jointloader, testloader, class_list, session):
    #     # Optimising only the projector
    #     if self.args.fine_tune_backbone_joint:
    #         optimizer = torch.optim.SGD([
    #             {'params': self.projector.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
    #     else:
    #         optimizer = self.get_optimizer_new(self.projector.parameters())

    #     # Setting up the scheduler
    #     if self.args.joint_schedule == "Milestone":
    #         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120],gamma=self.args.gamma)
    #     else:
    #         warmup_epochs = self.args.warmup_epochs_inc
    #         min_lr = 0.0
    #         scheduler = LinearWarmupCosineAnnealingLR(
    #                 optimizer, 
    #                 warmup_epochs=warmup_epochs, 
    #                 max_epochs=self.args.epochs_joint,
    #                 warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
    #                 eta_min=min_lr)

    #     sc_criterion = supcon.SupConLoss()
    #     pull_criterion = self.select_criterion()
    #     xent_criterion = nn.CrossEntropyLoss()

    #     best_loss = None
    #     best_hm = None
    #     best_acc = None
    #     best_projector = None
    #     best_l4 = None
    #     best_hm_epoch = 0

    #     hm_patience = self.args.hm_patience
    #     hm_patience_count = 0
        
    #     average_gap = Averager()
    #     average_gap_n = Averager()
    #     average_gap_b = Averager()
        
    #     if self.args.novel_bias:
    #         novel_class_start = self.args.base_class + (self.args.way * (session-1))
    #     elif self.args.queue_pull:
    #         queued_session = math.floor(session / 2)
    #         novel_class_start = self.args.base_class + (self.args.way * (queued_session))
    #     else:
    #         novel_class_start = self.args.base_class

    #     self.eval() # Fixing batch norm

    #     test_freq = self.args.testing_freq
        
    #     # Target prototypes contains assigned targets and unassigned targets (except base and current session targets)
    #     target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session)[self.args.base_class:].clone(), self.fc.rv.clone())))
    #     target_prototypes.requires_grad = False
    #     # target_labels = torch.arange(self.args.num_classes - self.args.base_class).cuda() + self.args.base_class
    #     target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class
    #     curr_labels = torch.arange(self.args.way * (session - 1), self.args.way * session).cuda() + self.args.base_class
    #     mask = ~torch.isin(target_labels, curr_labels)
    #     target_labels = target_labels[mask]

    #     # For simplex loss
    #     assigned_targets = self.fc.get_classifier_weights().detach().clone()
    #     assigned_targets_label = torch.arange(self.args.base_class + (self.args.way * session))
    #     unassigned_targets = self.fc.rv.detach().clone()    

    #     with torch.enable_grad():
    #         tqdm_gen = tqdm(range(self.args.epochs_joint))
    #         for epoch in tqdm_gen:
    #             sc_total_loss = 0
    #             pull_total_loss = 0
    #             ta = Averager()
    #             tl = {
    #                 "novel": Averager(),
    #                 "base": Averager()
    #             }
    #             for batch in jointloader:
    #                 images, label = batch
    #                 if torch.cuda.is_available():
    #                     images = [i.cuda(non_blocking=True) for i in images]
    #                     label = label.cuda(non_blocking=True)

    #                 bsz = label.shape[0]

    #                 pert_count = bsz
    #                 pert, target_labels = perturb_targets_norm_count(target_prototypes, target_labels, pert_count, 2, epsilon = self.args.perturb_epsilon_base)
    #                 perturbed_t1, perturbed_t2 = pert
    #                 projections, _, sc_loss = self.forward(images[0], im_k = images[1], labels=label,
    #                                                        perturbed_t1 = perturbed_t1, perturbed_t2 = perturbed_t2, pert_labels = target_labels)

    #                 # # Normalise projections
    #                 # projections = normalize(projections)

    #                 # f1, f2 = torch.split(projections, [bsz, bsz], dim=0)

    #                 # Add more labels for newer sessions
    #                 # new_ixs = torch.argwhere(label >= self.args.base_class).flatten()
    #                 # old_ixs = torch.argwhere(label < self.args.base_class).flatten()

    #                 loss = 0    

    #                 # Perturbing only unassigned and previous session targets and using cosine loss to pull novel classes together
    #                 # if self.args.skip_perturbation:
    #                 #     f1_ = f1
    #                 #     f2_ = f2
    #                 #     label_ = label
    #                 #     pert_count = 0
    #                 # else:
    #                 #     # import pdb; pdb.set_trace()
    #                 #     if self.args.remove_curr_features:
    #                 #         mask = ~torch.isin(label, curr_labels)
    #                 #         label_ = label[mask]
    #                 #         f1_ = f1[mask]
    #                 #         f2_ = f2[mask]
    #                 #     else:
    #                 #         f1_ = f1
    #                 #         f2_ = f2
    #                 #         label_ = label
                            
    #                 #     perturbed_t1 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_inc)
    #                 #     perturbed_t2 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_inc)
    #                 #     f1_ = torch.cat((f1_, perturbed_t1), axis = 0)
    #                 #     f2_ = torch.cat((f2_, perturbed_t2), axis = 0)
    #                 #     label_ = torch.cat((label_, target_labels))
    #                 #     pert_count = target_labels.shape[0] if self.args.remove_pert_numerator else 0
                        
    #                 # features = torch.cat([f1_.unsqueeze(1), f2_.unsqueeze(1)], dim=1)
    #                 # sc_loss = self.args.sup_lam * sc_criterion(features, label_, pert_count = pert_count)
    #                 # loss += sc_loss
    #                 loss += self.args.sup_lam * sc_loss

    #                 # Computing simplex loss
    #                 if self.args.simplex_lam_inc > 0:
    #                     sloss = simplex_loss(projections, label, assigned_targets, assigned_targets_label, unassigned_targets)
    #                     # sloss = simplex_loss(f1, label, assigned_targets, assigned_targets_label, unassigned_targets)
    #                     loss += self.args.simplex_lam_inc * sloss

    #                 # Cosine Pull Loss
    #                 logits = self.fc(projections)
    #                 pull_loss = self.args.cos_lam * self.pull_loss(label, novel_class_start, pull_criterion, logits, tl, session)
    #                 loss += pull_loss

    #                 ta.add(count_acc(logits, label))

    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()

    #                 sc_total_loss += sc_loss.item()
    #                 pull_total_loss += pull_loss.item()

    #             # Model Saving
    #             if epoch % test_freq == 0:
    #                 if self.args.validation_metric == "hm":
    #                     metrics = self.test_fc_head(self.fc, testloader, epoch, session)
    #                     # Validation
    #                     vhm = metrics["vhm"]
    #                     if best_hm is None or vhm > best_hm:
    #                         best_hm = vhm
    #                         best_hm_epoch = epoch
    #                         hm_patience_count = -1
    #                         best_projector = deepcopy(self.projector.state_dict())
    #                         if self.args.fine_tune_backbone_joint:
    #                             best_l4 = deepcopy(self.encoder.layer4.state_dict())
    #                     out_string = '(Joint) Sess: {}, loss: sc/pull {:.3f}/{:.3f}|(b/n)({:.3f}/{:.3f}), trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (test) b/n={:.3f}/{:.3f}, bb/nn={:.3f}/{:.3f}'\
    #                         .format(
    #                                 session, 
    #                                 sc_total_loss,
    #                                 pull_total_loss,
    #                                 float('%.3f' % (tl["base"].item())),
    #                                 float('%.3f' % (tl["novel"].item())),
    #                                 float('%.3f' % (ta.item() * 100.0)),
    #                                 float("%.3f" % (metrics["va"] * 100.0)),
    #                                 float("%.3f" % (best_hm * 100.0)),
    #                                 float("%.3f" % (metrics["vaBase"] * 100.0)),
    #                                 float("%.3f" % (metrics["vaNovel"] * 100.0)),
    #                                 float("%.3f" % (metrics["vaBaseB"] * 100.0)),
    #                                 float("%.3f" % (metrics["vaNovelN"] * 100.0)),
    #                             )
    #                     fp = metrics["fpr"]
    #                     out_string += f", (False Positives) n2b|b2n|n2n|b2b:{fp['novel2base']:.3f}|{fp['base2novel']:.3f}|{fp['novel2novel']:.3f}|{fp['base2base']:.3f}"
    #                     hm_patience_count += 1

    #             tqdm_gen.set_description(out_string)

    #             if hm_patience_count > hm_patience:
    #                 break
                
    #             scheduler.step()

    #     print("Best HM found at epoch: ", best_hm_epoch)
        
    #     self.projector.load_state_dict(best_projector, strict=True)
    #     if self.args.fine_tune_backbone_joint:
    #         self.encoder.layer4.load_state_dict(best_l4, strict = True)        

    def update_fc_ft_joint_supcon(self, jointloader, testloader, class_list, session):
        # Optimising only the projector
        if self.args.fine_tune_backbone_joint:
            optimizer = torch.optim.SGD([
                {'params': self.projector.parameters()},
                {"params": self.predictor.parameters()},
                {'params': self.encoder.parameters(), 'lr': 5e-4}
            ], lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        else:
            optimized_parameters = [
                {"params": self.projector.parameters()},
                {"params": self.predictor.parameters()}
                ]
            optimizer = self.get_optimizer_new(optimized_parameters)

        # Setting up the scheduler
        if self.args.joint_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120],gamma=self.args.gamma)
        else:
            warmup_epochs = self.args.warmup_epochs_inc
            min_lr = 0.0
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

        self.eval() # Fixing batch norm

        test_freq = self.args.testing_freq
        
        # target_prototypes = torch.nn.Parameter(self.fc.get_classifier_weights())
        # target_labels = torch.arange(self.args.base_class + (self.args.way * session)).cuda()

        # target_prototypes = torch.nn.Parameter(self.fc.rv)
        # target_labels = torch.arange(self.args.num_classes - (self.args.base_class + (self.args.way * session)) ).cuda() + (self.args.base_class + (self.args.way * session))

        # Target prototypes contains assigned targets and unassigned targets (except base and current session targets)
        target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session)[self.args.base_class:].clone(), self.fc.rv.clone())))
        target_prototypes.requires_grad = False
        # target_labels = torch.arange(self.args.num_classes - self.args.base_class).cuda() + self.args.base_class
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class
        curr_labels = torch.arange(self.args.way * (session - 1), self.args.way * session).cuda() + self.args.base_class
        mask = ~torch.isin(target_labels, curr_labels)
        target_labels = target_labels[mask]

        # For simplex loss
        assigned_targets = self.fc.get_classifier_weights().detach().clone()
        assigned_targets_label = torch.arange(self.args.base_class + (self.args.way * session))
        unassigned_targets = self.fc.rv.detach().clone()

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
                            
                        perturbed_t1 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_inc)
                        perturbed_t2 = perturb_targets_norm(target_prototypes, epsilon = self.args.perturb_epsilon_inc)
                        f1_ = torch.cat((f1_, perturbed_t1), axis = 0)
                        f2_ = torch.cat((f2_, perturbed_t2), axis = 0)
                        label_ = torch.cat((label_, target_labels))
                        pert_count = target_labels.shape[0] if self.args.remove_pert_numerator else 0
                        # pert_count = 0
                        
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

                    # Cosine Pull Loss
                    logits = self.fc(self.predictor(encodings))
                    label_rep = label.repeat(2)
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
                if epoch % test_freq == 0:
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
                        out_string = '(Joint) Sess: {}, loss: sc/pull {:.3f}/{:.3f}|(b/n)({:.3f}/{:.3f}), trainAcc {:.3f}, testAcc {:.3f}, bestHM {:.3f}, (test) b/n={:.3f}/{:.3f}, bb/nn={:.3f}/{:.3f}'\
                            .format(
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

                encoding = self.predictor(self.encode(data, return_encodings=True)[1].detach())

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