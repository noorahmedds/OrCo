"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
# """
    
    
from __future__ import print_function

import torch
import torch.nn as nn

import math
import numpy as np


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm_sup_con and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr_sup_con
    if args.cosine_sup_con:
        eta_min = lr * (args.lr_decay_rate_sup_con ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs_sup_con)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs_sup_con))
        if steps > 0:
            lr = lr * (args.lr_decay_rate_sup_con ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, lam_at = 0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.lam_at = lam_at

    def forward(self, features, labels=None, mask=None, margin = 0, hemisphere = False, target_prototypes = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """ 
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),   # Inner product (in contrast mode all: the inner product is between all features if "one" then only the current feature with the augmentations and others in batch)
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # logits now contain the dot product of the anchors with each feature

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases (i.e. in the mask replace 1's with 0's in the position of the anchor in the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask   # in the original mask we now suppress self contrast cases augmented anchor class

        # Subtract margin from the logits of positive cases note anchor dot contrast has self contrast in the diagonal
        logits[mask > 0] -= margin

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask    # Exponential applied to all logits except the anchor one
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        at_loss = 0
        # if target_prototypes is not None:
        #     curr_batch_feature = anchor_feature
        #     # atd = torch.div(torch.matmul(curr_batch_feature, target_prototypes.T), self.temperature) # achnor_traget_dot
        #     atd = torch.matmul(curr_batch_feature, target_prototypes.T) # achnor_traget_dot
        #     vu = torch.cat((target_prototypes, anchor_feature))
        #     den = torch.matmul(curr_batch_feature, vu.T)

        #     at_logits_max, _ = torch.max(atd, dim=1, keepdim=True)
        #     at_logits = atd - at_logits_max.detach()  # logits now contain the dot product of the anchors with each feature
            
        #     at_mask = torch.eq(labels, torch.arange(target_prototypes.shape[0]).cuda()).float().to(device)
        #     at_mask = at_mask.repeat(2, 1)
        #     at_exp_logits = torch.exp(at_logits) * at_mask   # get only the angles between targets and embeddings
        #     # at_log_prob = at_logits - torch.log(at_exp_logits.sum(1, keepdim=True))

        #     at_mask_den = torch.eq(labels.repeat(2,1), torch.cat((torch.arange(target_prototypes.shape[0]).cuda(), labels.flatten().repeat(2)))).float().to(device)
        #     at_exp_den = torch.exp(den) * at_mask_den

        #     at_log_probs = torch.log(at_exp_logits.sum(1, keepdim=True)) - torch.log(at_exp_den.sum(1, keepdim=True))
        #     at_mean_log_prob_pos = (at_mask * at_log_probs).sum(1) / at_mask.sum(1)
        #     at_loss = - (1 / 0.02) * at_mean_log_prob_pos # 0.05 works
        #     at_loss = self.lam_at * at_loss.view(anchor_count, batch_size).mean()

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss + at_loss
    

def find_eq_vector(vector_as_rows):
    A = (vector_as_rows[0] - vector_as_rows[1:]) * 2
    res = torch.matmul(torch.linalg.pinv(A), (vector_as_rows[0]**2 - vector_as_rows[1:]**2).sum(axis=1))

    return res

def find_equidistant_vectors(vectors, skip_encode_norm = False):
    # Calculate the center of the hypersphere
    center = torch.mean(vectors, dim=0)

    # Determine the radius of the hypersphere
    radius = torch.norm(vectors[0] - center)

    # Find the equatorial hyperplane
    radius_vector = vectors[0] - center
    equatorial_hyperplane = torch.nn.functional.normalize(radius_vector, dim=0)

    equidistant_vectors = []

    # Calculate equidistant vectors
    for vector in vectors:
        # Project the vector onto the equatorial hyperplane
        projection = vector - torch.dot(vector - center, equatorial_hyperplane) * equatorial_hyperplane

        # Apply rotation around the radius vector
        rotated_vector = center + 2 * (projection - center)

        equidistant_vectors.append(rotated_vector)
    out = torch.stack(equidistant_vectors)
    # Normalizing the reserve vectors
    
    if skip_encode_norm:
       return out
    
    return torch.nn.functional.normalize(out, dim = 1) 

def find_equiangular_vectors(existing_vectors, n=100, d=512, iterations=100):
    # Combine existing vectors with newly generated vectors
    existing_vectors = existing_vectors.detach().cpu().numpy()
    m = existing_vectors.shape[0]
    rand_vec = np.random.randn(n, d)
    rand_vec = rand_vec/np.linalg.norm(rand_vec, axis = 1, keepdims=True)

    vectors = np.concatenate((existing_vectors, rand_vec))

    # Step 4: Optimization loop
    for _ in range(iterations):
        # Step 4a: Calculate pairwise angles
        dot_products = np.abs(np.dot(vectors, vectors.T)) - np.finfo(float).eps
        angles = np.arccos(dot_products)

        # Step 4b: Compute centroids
        centroids = np.mean(vectors, axis=0)

        # Step 4c: Move vectors towards centroids
        vectors[m:] += (centroids - vectors[m:])

        # Step 1: Normalize vectors
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    return torch.tensor(vectors[m:]).cuda()