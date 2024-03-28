import os
import pprint as pprint
import random
import time

import numpy as np
import torch

_utils_pp = pprint.PrettyPrinter()

from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

import torch.nn.functional as F


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    return orth_vec

def pprint(x):
    _utils_pp.pprint(x)

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def set_layers(args):
    layer_choice = [int(x) for x in args.mixup_layer_choice.split(',')]
    return layer_choice

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        if x is None:   # Skipping addition of empty items
            return
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():      
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def hm(a,b):
    try:
        return ((2 * a * b) / (a + b))
    except ZeroDivisionError:
        print("HM (ZeroDivision Error)")
        return 0

def am(a,b):
    return (a+b) / 2

def createSessionalConfusionMatrix(y_true, y_pred, classes, cmap = None, hline_at = None, vline_at= None, summarize = False, session = 0): #cmap="crest"
    # Build confusion matrix "Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class."

    session_ixs = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    session_ixs = session_ixs[:session+1]

    # Confusion matrix gt vs preds. if we want to see which preds are 
    # For all sessions sum the predictions up
    # Map each label to its session
    y_true = np.digitize(y_true, session_ixs)
    y_pred = np.digitize(y_pred, session_ixs)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/cf_matrix.sum(axis = 1), index=[i for i in session_ixs],
                         columns=[i for i in session_ixs])

    # Compute for each class the top 5 most confused predictions
    top_preds = np.flip(np.argsort(df_cm.to_numpy(), axis = 1)[:, -6:], axis = 1)

    plt.figure(figsize=(12, 7))    
    heatmap = sn.heatmap(df_cm, annot=False, cmap = cmap, vmin=0, vmax=1)
    if hline_at is not None:
        line = heatmap.hlines([1],*heatmap.get_xlim(), colors='y')
        line.set_alpha(0.7)
    if vline_at is not None:
        line = heatmap.vlines([1],*heatmap.get_ylim(), colors='y')
        line.set_alpha(0.7)


    y_ticks_indices = np.arange(0, len(session_ixs), 1)
    heatmap.set_yticks(y_ticks_indices)
    heatmap.set_yticklabels([session_ixs[i] for i in y_ticks_indices])

    x_ticks_indices = np.arange(0, len(session_ixs), 1)
    heatmap.set_xticks(x_ticks_indices)
    heatmap.set_xticklabels([session_ixs[i] for i in x_ticks_indices])

    plt.yticks(rotation=0)
    heatmap.set(xlabel="Predicted Classes", ylabel="True Classes", title=f"Session: {session}")
    return {"cm": heatmap.get_figure(), "df":df_cm}

def normalize(x): 
    x = x / torch.norm(x, dim=1, p=2, keepdim=True)
    return x

import math
def perturb_targets_norm_count(targets, target_labels, ncount, nviews, epsilon = 1, offset = 0):
    # Create ncount perturbations
    # Output should have each target represented
    # Return the labels as well
    
    views = []
    ix = torch.randperm(targets.shape[0])
    if ix.shape[0] < ncount:
        rep_count = math.ceil(ncount/ix.shape[0])            
        ix = ix.repeat(rep_count)[:ncount]
        ix = ix[torch.randperm(ix.shape[0])]
    else:
        ix = ix[:ncount]
    for v in range(nviews):
        rand = ((torch.rand(ncount, targets.shape[1]) - offset) * epsilon).to(targets.device)
        views.append(normalize(targets[ix] + rand))
    
    target_labels = target_labels[ix]
    return views, target_labels


def simplex_loss(feat, labels, assigned_targets, assigned_targets_label, unassigned_targets):
    # === Apply on averaged features + all targets (assign not in batch + unassigned)
    unique_labels, inverse_indices, counts = torch.unique(labels, return_inverse=True, return_counts = True)
    averaged = torch.zeros(len(unique_labels), feat.shape[1]).cuda()
    
    for i, l in enumerate(unique_labels):
        label_indices = torch.where(labels == l)[0]
        averaged_row = torch.mean(feat[label_indices], dim=0)
        averaged[i] = averaged_row
    averaged = normalize(averaged)

    # Average assigned targets + the averaged batch
    mask = ~torch.isin(assigned_targets_label.cuda(), unique_labels) # assigned targets not in batch
    assigned_targets_not_in_batch = assigned_targets[mask]
    all_targets = normalize(torch.cat((averaged, assigned_targets_not_in_batch, unassigned_targets))) # original

    sim = F.cosine_similarity(all_targets[None,:,:], all_targets[:,None,:], dim=-1)
    loss = torch.log(torch.exp(sim/1).sum(axis = 1)).sum() / all_targets.shape[0]

    return loss
