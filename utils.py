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

class MultipleOptimizer(torch.optim.Optimizer):
    def __init__(self, *op):
        self.optimizers = op

        self.param_groups = []
        for o in self.optimizers:
            self.param_groups.extend(o.param_groups)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def set_up_sweep_args(args, wandb_configs):
    # Set args from the sweep configs
    for k in wandb_configs.keys():
        v = wandb_configs[k]
        if str(v) == "true":
            setattr(args, k, True)
        elif str(v) == "false":
            setattr(args, k, False)
        else:
            setattr(args, k, v)

    return args

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
    # import pdb; pdb.set_trace()
    # assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
    #     "The max irregular value is : {}".format(
    #         torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
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

import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from copy import deepcopy

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, data_loader, way):

        self.model = model
        self.dataset = data_loader  # older dataset.
        self.way = way

        # In our case the dataset will be the set of old exemplars. i.e. B and n(i-1). 

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()

        for input in self.dataset:
            self.model.zero_grad()
            data, label = input
            data = variable(data)

            # Pass through the entire model including model.fc. 
            output, enc = self.model(data)

            # Remove output logits corresponding to the newest task/session
            output = output.view(1, -1)[:, :-self.way]
            
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if "encoder" in n:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

        # importance * ewc.penalty(model) where you would like to add this loss


def mixup_data_sup_con(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda for sup con data. i.e. 2 views concatenated'''
    # So we mixup in the first half the same as we mixup in the second half of the tensor 
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]//2
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        index_feature = index.repeat(2)
    else:
        index = torch.randperm(batch_size)
        index_feature = index.repeat(2)

    mixed_x = lam * x + (1 - lam) * x[index_feature, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def compute_hn_cosine_embedding_loss(encoding, logits, label, session):
    # Get softmax weighting for hardnegatives

    # All targets must be -1

    # For each sample in the batch i.e. encoding.shape[0]
    # Compute the softmax weighting
    # Now from all the encodings, get the encodings which dont have the same label
    # To these encodings also add fc.classifiers which dont belong to the current class
    # Combine encoding + fc.classifiers/current
    # xi matmul with the combined classifier
    # now compute the 

    # Compute hardnegative logits
    hn_scores = F.softmax(logits, dim = 1)[:, label]

    # Get Cosine scores between logits
    n_enc = normalize(encoding)
    cos_scores = torch.matmul(n_enc, n_enc.T).clip(min=0)   # - margin

    # Remove all cos scores where the row and column label are the same
    mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    cos_scores.masked_fill_(mask, 0)

    # Multipy hard negative scores with cosine scores to 
    loss = torch.mul(hn_scores, cos_scores).sum()
    
    return loss




def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():      
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

import itertools
def count_acc_sub(logits, label, session, args, base_t, novel_t):
    test_class = args.base_class + session * args.way
    m = torch.nn.Softmax(dim = 1)
    novel_start_id = test_class - args.way
    logits = logits[:, :test_class] # Limiting the scope of the current session to the logits of the activate classes
    base_mask = label < args.base_class
    
    # novel_mask = label >= novel_start_id <<
    novel_mask = label >= args.base_class       # The novel mask masks all classes which are from the novel sessions. Not necessarily the ones from the current session

    if torch.cuda.is_available():
        # Performance of novel samples on joint space
        logits[:, :args.base_class] = logits[:, :args.base_class] / base_t
        logits[:, args.base_class:] = logits[:, args.base_class:] / novel_t

        # pred = torch.argmax(m(logits), dim=1)[novel_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        pred = torch.argmax(logits, dim=1)[novel_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        label_ = label[novel_mask]
        novel_acc = (pred == label_).type(torch.cuda.FloatTensor)
        if novel_acc.numel() == 0:
            novel_acc = None
        else:
            novel_acc = novel_acc.mean().item()
        

        # pred = torch.argmax(m(logits), dim=1)[base_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        pred = torch.argmax(logits, dim=1)[base_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        label_ = label[base_mask]
        base_acc = (pred == label_).type(torch.cuda.FloatTensor)
        if base_acc.numel() == 0:
            base_acc = None
        else:
            base_acc = base_acc.mean().item()

        return novel_acc, base_acc
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_max(logits, label, session, args, base_t, novel_t):
    """
        Get the best label from each session 0, 1...n-1, n where n is the current session if n>1 else 
        get best pred label from session 0, 1 or 0
    """    
    base_end_ix = args.base_class
    novel_start_ix = args.base_class + ((session - 1) * args.way)

    # Apply softmax here: And try with temperature as well. Try on the test and validation set. 
    m = torch.nn.Softmax(dim = 1)

    # Temperature variables for scaling
    # base_t = 0.005 # temperature low for higher average confidence
    # novel_t = 0.2 # temperature high for lower average confidence
    
    base_probs =   m(logits[:, :base_end_ix]/base_t)
    novel_probs =  m(logits[:, args.base_class:novel_start_ix + args.way]/novel_t) # Now taking the entire novel classe up until the class that is in consideration
    
    # if novel_start_ix == base_end_ix:
    #     interm_probs = torch.zeros_like(novel_probs)
    # else:
    #     interm_probs = m(logits[:, base_end_ix:novel_start_ix]/novel_t)

    # For each indivudal classifier we find the max softmax scores
    base_best = torch.max(base_probs, dim=1)
    novel_best = torch.max(novel_probs, dim=1)
    # interm_best = torch.max(interm_probs, dim=1)

    # TODO: Save softmax output for base_probs for a single class. And same class show the novel_best and interm best
    # if novel_start_ix in label:import pdb; pdb.set_trace()
    # base_numpy = novel_probs.mean(axis = 0).cpu().numpy()
    # fig1 = plt.figure()
    # ax1 = fig.add_subplot(111)
    # hist, bins = np.histogram(base_numpy, bins=8)
    # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))
    # plt.hist(base_numpy, bins=logbins)
    # plt.xscale('log')
    # plt.savefig("./base_probs.png")

    # Finding the indices of the samples where the classifier performs better for the individual classifier
    # So to say which classifier is more confident for which sample
    # better_base_mask = torch.logical_and((base_best.values >= novel_best.values), (base_best.values >= interm_best.values))
    # better_novel_mask = torch.logical_and((novel_best.values >= base_best.values), (novel_best.values >= interm_best.values))
    # better_interm_mask = torch.logical_and((interm_best.values >= base_best.values), (interm_best.values >= novel_best.values))
    better_base_mask = base_best.values >= novel_best.values
    better_novel_mask = novel_best.values > base_best.values

    # Finding the accuracy for the idnividual classifier given the samples it is more confident on.
    if label[better_base_mask].shape[0] != 0:
        acc_base = (base_best.indices[better_base_mask] == label[better_base_mask]).type(torch.cuda.FloatTensor)
    else:
        acc_base = None
    if label[better_base_mask].shape[0] != 0:
        acc_novel = (novel_best.indices[better_novel_mask] + args.base_class  == label[better_novel_mask]).type(torch.cuda.FloatTensor)
    else:
        acc_novel = None

    total_acc = torch.cat([acc_base, acc_novel]).mean()
    acc_base = acc_base.mean().item()
    acc_novel = acc_novel.mean().item()

    # acc_interm = (interm_best.indices[better_interm_mask] + args.base_class  == label[better_interm_mask]).type(torch.cuda.FloatTensor).sum()

    # total_acc = 0
    # count = 0
    # if not torch.isnan(acc_base):
    #     total_acc += acc_base.item()
    # if not torch.isnan(acc_novel):
    #     total_acc += acc_novel.item()
    # if not torch.isnan(acc_interm):
    #     total_acc += acc_interm.item()

    # if best_acc is None:
    #     best_acc = total_hm
    #     best_comb = c
    # else:
    #     if total_hm > best_acc:
    #         best_acc = total_hm
    #         best_comb = c

    # print("Best Accuracy: ", best_acc)
    # print("Best Temperature Combination: ", best_comb)

    return acc_base, acc_novel, total_acc

def count_acc_previous(logits, label, test_class, args, sub_space = "joint"):
    novel_start_id = test_class - (2*args.way)
    novel_end_id = test_class - args.way
    logits = logits[:, :test_class] # Limiting the scope of the current session to the logits of the activate classes
    
    # novel_mask = label >= novel_start_id <<
    prev_novel_mask = torch.logical_and(label >= args.base_class, label <= novel_end_id)

    if torch.cuda.is_available():
        # Performance of (all) novel samples on joint space.
        pred = torch.argmax(logits, dim=1)[prev_novel_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        
        label_ = label[prev_novel_mask]
        novel_acc = (pred == label_).type(torch.cuda.FloatTensor)

        if novel_acc.numel() == 0:
            novel_acc = None
        else:
            novel_acc = novel_acc.mean().item()
        
        return novel_acc
    else:
        raise Exception
        
def compute_geometric_mean(tensor):
    geometric_means = torch.prod(tensor, dim=0) ** (1.0 / tensor.shape[0])
    return geometric_means

def count_acc_binary(logits, label, test_class, args, sub_space = "joint"):
    novel_start_id = test_class - args.way
    logits = logits[:, :test_class] # Limiting the scope of the current session to the logits of the activate classes
    base_mask = label < args.base_class
    novel_mask = label >= args.base_class

    pred = torch.argmax(logits, dim=1)[novel_mask]
    novel_correct = (pred >= args.base_class).type(torch.cuda.FloatTensor).sum()

    pred = torch.argmax(logits, dim=1)[base_mask]
    base_correct = (pred < args.base_class).type(torch.cuda.FloatTensor).sum()

    binary_acc = (novel_correct + base_correct) / (base_mask.sum() + novel_mask.sum())

    return binary_acc


def count_fp(logits, label, test_class, args, fpr):
    logits = logits[:, :test_class] # Limiting the scope of the current session to the logits of the activate classes

    base_mask = label < args.base_class
    novel_mask = label >= args.base_class

    novel_count = novel_mask.type(torch.cuda.FloatTensor).sum().item()
    fpr["total_novel"] += novel_count
    fpr["total_base"] += base_mask.type(torch.cuda.FloatTensor).sum().item()

    if novel_count > 0:
        pred = torch.argmax(logits, dim=1)[novel_mask]
        novel_correct = (pred >= args.base_class).type(torch.cuda.FloatTensor).sum().item()
        fpr["novel2base"] += (pred < args.base_class).type(torch.cuda.FloatTensor).sum().item()   # Novel classes predicted as base

        pred = torch.argmax(logits[:, args.base_class:], dim=1)[novel_mask]
        pred += args.base_class
        fpr["novel2novel"] += (pred != label[novel_mask]).type(torch.cuda.FloatTensor).sum().item()

    pred = torch.argmax(logits, dim=1)[base_mask]
    base_correct = (pred < args.base_class).type(torch.cuda.FloatTensor).sum().item()
    fpr["base2novel"] += (pred >= args.base_class).type(torch.cuda.FloatTensor).sum().item()

    pred = torch.argmax(logits[:, :args.base_class], dim=1)[base_mask]
    fpr["base2base"] += (pred != label[base_mask]).type(torch.cuda.FloatTensor).sum().item()

    return fpr


def count_acc_(logits, label, test_class, args, sub_space = "joint"):
    novel_start_id = test_class - args.way
    logits = logits[:, :test_class] # Limiting the scope of the current session to the logits of the activate classes
    base_mask = label < args.base_class
    
    # novel_mask = label >= novel_start_id <<
    novel_mask = label >= args.base_class       # The novel mask masks all classes which are from the novel sessions. Not necessarily the ones from the current session

    if torch.cuda.is_available():
        # Performance of (all) novel samples on joint space.
        if sub_space == "joint":
            pred = torch.argmax(logits, dim=1)[novel_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        else:
            pred = torch.argmax(logits[:, args.base_class:], dim=1)[novel_mask]
            pred += args.base_class
            # TODO: This needs to be fixed. The predictions need to be added with an offset
        label_ = label[novel_mask]
        novel_acc = (pred == label_).type(torch.cuda.FloatTensor)
        if novel_acc.numel() == 0:
            novel_acc = None
        else:
            novel_acc = novel_acc.mean().item()
        
        if sub_space == "joint":
            pred = torch.argmax(logits, dim=1)[base_mask]  # Taking argmax in the joint space. i.e. novel + all previous classes. Todo: It is possible that a novel class from a previous section may over power the novel class from the new session and this should be ameliorated
        else:
            pred = torch.argmax(logits[:, :args.base_class], dim=1)[base_mask]
        label_ = label[base_mask]
        base_acc = (pred == label_).type(torch.cuda.FloatTensor)
        if base_acc.numel() == 0:
            base_acc = None
        else:
            base_acc = base_acc.mean().item()

        return novel_acc, base_acc
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_session(logits, label, args):
    acc_list = []
    for sess in range(args.sessions):
        if sess == 0:
            mask = label < args.base_class
        else:
            mask = torch.logical_and(label >= args.base_class + (args.way * (sess - 1)), label < args.base_class + (args.way*sess))
        pred = torch.argmax(logits, dim=1)[mask]
        label_ = label[mask]

        if label_.numel() == 0:
            break

        acc_list.append((pred == label_).type(torch.cuda.FloatTensor).mean().item())

    return acc_list
    


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def split_logits(logits, session, base_class, way):
    
    return logits[:base_class], logits[base_class + (way * session-1):] 

def harmonic_mean(data):
    # data is a list type
    num = len(data)
    den = 0.0
    for i in data:
        if i == 0:
            return 0
        den += 1/i

    return num/den

def hm(a,b):
    try:
        return ((2 * a * b) / (a + b))
    except ZeroDivisionError:
        print("HM (ZeroDivision Error)")
        return 0

def hm3(a,b,c):
    try:
        return (3 * a * b * c) / (a * b + b * c + c * a)
    except Exception as e:
        return 0.0

def am(a,b):
    return (a+b) / 2

def appendSingleTrainingExemplars(dataset, args):
    """
        Take only labels from args.base_class and in data self.data append the single exemplar
    """
    label2data = {}
    for k,v in dataset.data2label.items():
        if v < args.base_class:
            if v not in label2data: label2data[v] = []
            label2data[v].append(k)
    
    n_samples = len(label2data[0])
    sample_ix = torch.randint(n_samples, (args.base_class,))

    data_tmp = []
    targets_tmp = []

    for i in range(args.base_class):
        data_tmp.append(label2data[i][sample_ix[i]])
        targets_tmp.append(i)

    dataset.data.extend(data_tmp)
    dataset.targets.extend(targets_tmp)

def bn_cross_entropy(preds, targets, args, session, lymbda=1):
    """
        Expects input parameters "preds" of type dict {"base_logits":[Tensor], "novel_logits":[Tensor]}
    """

    base_end_ix = args.base_class
    novel_start_ix = args.base_class + ((session - 1) * args.way)

    base_logits =   preds[:, :base_end_ix]
    interm_logits = preds[:, base_end_ix:novel_start_ix]
    novel_logits =  preds[:, novel_start_ix:novel_start_ix + args.way]

    # From the gist at: https://gist.github.com/yang-zhang/217dcc6ae9171d7a46ce42e215c1fee0
    s = novel_logits.exp() / (novel_logits.exp().sum() + base_logits.exp().sum()).unsqueeze(-1)
    
    # filler = torch.zeros(novel_logits.shape[0], targets.min()).cuda()
    s = torch.hstack((base_logits, interm_logits, s))   # s.shape == torch.Size([25, 60])
    
    loss = -s[range(targets.shape[0]), targets].log().mean()
    
    return loss

def createConfusionMatrix(y_true, y_pred, classes, cmap = None, hline_at = None, vline_at= None, summarize = False, session = 0): #cmap="crest"
    # Build confusion matrix "Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class."
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Confusion matrix gt vs preds. if we want to see which preds are 
    if summarize:
        cf_matrix_summ = np.zeros((2,2))
        cf_matrix_summ[0,0] = cf_matrix[:hline_at, :vline_at].sum()
        cf_matrix_summ[0,1] = cf_matrix[:hline_at, vline_at:].sum()
        cf_matrix_summ[1,0] = cf_matrix[hline_at:, :vline_at].sum()
        cf_matrix_summ[1,1] = cf_matrix[hline_at:, vline_at:].sum()
        classes = ["base", "novel"]
        cf_matrix = cf_matrix_summ

    # Normalise the confusion matrix. Note that in the test set each class does not have equal number of samples
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
    #                      columns=[i for i in classes])

    df_cm = pd.DataFrame(cf_matrix/cf_matrix.sum(axis = 1), index=[i for i in classes],
                         columns=[i for i in classes])

    # Compute for each class the top 5 most confused predictions
    top_preds = np.flip(np.argsort(df_cm.to_numpy(), axis = 1)[:, -6:], axis = 1)

    plt.figure(figsize=(12, 7))    
    heatmap = sn.heatmap(df_cm, annot=False, cmap = cmap, vmin=0, vmax=1)
    if hline_at is not None:
        # line = heatmap.hlines([hline_at], *heatmap.get_xlim(), colors='y')
        # line.set_alpha(hline_alpha)  # Set opacity of the horizontal line
        line = heatmap.hlines([hline_at],*heatmap.get_xlim(), colors='y')
        line.set_alpha(0.7)
    if vline_at is not None:
        line = heatmap.vlines([vline_at],*heatmap.get_ylim(), colors='y')
        line.set_alpha(0.7)

    y_ticks_indices = np.arange(0, len(classes), 10)
    heatmap.set_yticks(y_ticks_indices)
    heatmap.set_yticklabels([classes[i] for i in y_ticks_indices])

    x_ticks_indices = np.arange(0, len(classes), 10)
    heatmap.set_xticks(x_ticks_indices)
    heatmap.set_xticklabels([classes[i] for i in x_ticks_indices])

    plt.yticks(rotation=0)
    heatmap.set(xlabel="Predicted Classes", ylabel="True Classes", title=f"Session: {session}")


    return {"cm": heatmap.get_figure(), "df":df_cm}

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

def KP_loss(theta1, theta2, lymbda_kp=1e-4):
    """
        Theta1 and 2 are state dictionaries of the models from phase 1 and 2 respectively
        TODO: Another parameter here could be the number of resent layers that will be used for the weight constraint calculation
    """
    loss = None

    for name, param in theta2.named_parameters():
        param_l2 = (param - theta1[name]).pow(2).sum()
        loss = param_l2 if loss is None else loss + param_l2


    # TODO: https://github.com/Annusha/LCwoF/blob/f83dad1544b342323e33ea51f17bc03650e1e694/mini_imgnet/resnet12.py#L141 implement this to be able to adjust which layers are skipped and which are not

    assert loss is not None, print("Error computing loss for the Knowledge Preservation module")

    return loss * lymbda_kp

def compute_parameter_diff(theta1, theta2):
    diff = 0
    for name, param in theta2.named_parameters():
        param_abs_diff = (param - theta1[name]).abs().sum().detach()
        diff = diff + param_abs_diff
    return diff

def infoNCE(fc):
    # Traverse all indiviudal prototypes of the fully connected layer
    # Traverse individual prototypes
    
    innerprod_mat = torch.matmul(fc, fc.T)
    x = torch.arange(innerprod_mat.shape[0])
    innerprod_mat[x, x] = 0
    F = innerprod_mat.mean()    # We want to minimise this number
    F = -torch.log(1.0/F)
    return F

def dot_loss(output, label, cur_M, classifier, criterion, H_length, reg_lam=0):
    # From https://github.com/NeuralCollapseApplications/ImbalancedLearning/blob/main/train.py
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss

def compute_dino_loss(student_output, teacher_output):
    # compute loss between the student and teacher output i.e. the dot between the mlp network feature space on to the fixed classifier
    
    # Student output contains in a tuple the logit scores p1. Number of crops are determined by the ncrops variable
    # Teacher output contains in a tuple the cross entropy scores p2 detached

    # For each crop create the new 
    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_output):
        for v in range(len(student_output)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_output[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms
    return total_loss

# def visualise_dino_features():
    # Data loader load
    # load model
    # Inference on base classes 
    # Save feature outputs over the encoder or decide where to take it from

    # Display feature outputs in plot


def adjust_learning_rate(optimizer, epoch, total_epochs, lr, decay_rate = 0.1, strategy="cosine"):
    if strategy=="cosine":
        eta_min = lr * (decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / total_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(decay_rate))
        if steps > 0:
            lr = lr * (args.lr_decay_rate_sup_con ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def normalize(x): 
    x = x / torch.norm(x, dim=1, p=2, keepdim=True)
    return x

def radial_label_smoothing(criterion, logits, label, target, alpha=0.0):
    # label = (label * (1-alpha)) + (logits * alpha/label.shape[1])
    label = (label * (1-alpha)) + (alpha/label.shape[1])
    label = normalize(label)
    loss = criterion(logits, label, target)
    return loss

def compute_off_element_mean(matrix):
    num_rows, num_cols = matrix.shape
    ones_matrix = np.ones((num_rows, num_cols))
    excluded_row_col_mask = 1 - np.eye(num_rows, num_cols)
    excluded_row_col_mask = excluded_row_col_mask[:, :, np.newaxis, np.newaxis]

    # sub_matrix_sum = np.sum(matrix * ones_matrix * excluded_row_col_mask, axis=(1, 2))
    # count = np.sum(ones_matrix * excluded_row_col_mask, axis=(1, 2))
    # result_matrix = sub_matrix_sum / count
    # return result_matrix
    sub_matrix_max = np.max(matrix * ones_matrix * excluded_row_col_mask, axis=2).mean(axis = 1)
    sum_of_max_in_submatrix = np.sum(sub_matrix_max)
    return sum_of_max_in_submatrix/2

from scipy.spatial.distance import mahalanobis
def compute_pairwise_mahalanobis(base_prototypes, base_cov, rv):
    # base cov is the covariance matrix for each base prototype
    pairwise_distances = np.zeros((base_prototypes.shape[0], rv.shape[0]))
    for i, row1 in enumerate(base_prototypes):
        for j, row2 in enumerate(rv):
            pairwise_distances[i, j] = mahalanobis(row1, row2, base_cov[i])

    return pairwise_distances

def perturb_targets_norm(targets, epsilon = 1):
    # Sample from gaussian distribution
    rand = torch.rand_like(targets) * epsilon
    perturbed_targets = targets + rand
    return normalize(perturbed_targets)

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

def perturb_targets_norm_count_gaus(targets, target_labels, ncount, nviews, epsilon = 1):
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
        rand = torch.empty(ncount,targets.shape[1]).normal_(mean=0, std=epsilon).to(targets.device)
        views.append(normalize(targets[ix] + rand))
    
    target_labels = target_labels[ix]
    return views, target_labels

def perturb_targets_w_variance(targets, average_class_variance, generated_instances = 1):
    # for each target the class variance is basically the class variance in the original distribution of the class on the hypersphere
    num_features = targets.shape[1]
    perturbed_targets = torch.zeros_like(targets)

    for k in range(targets.shape[0]):
        perturbed_targets[k] = torch.tensor(np.random.multivariate_normal(targets[k].detach().cpu().numpy(),  np.eye(num_features) * average_class_variance.T, 1))
        
    return normalize(perturbed_targets)


def interpolate_target(targets, samples, alpha = 0.5):
    # Find an interpolant vector between the target and samples
    # Ensure tensors have the same shape
    assert targets.shape == samples.shape, "Tensors must have the same shape"
    
    # Interpolate between the two tensors
    interpolated_tensor = (1 - alpha) * targets + alpha * samples
    
    return normalize(interpolated_tensor)

def simplex_loss_in_batch(feat):
    sim = F.cosine_similarity(feat[None,:,:], feat[:,None,:], dim=-1)
    loss = torch.log(torch.exp(sim/1).sum(axis = 1)).sum() / feat.shape[0]
    return loss

def simplex_loss(feat, labels, assigned_targets, assigned_targets_label, unassigned_targets):
    # Normalise feat and all targets
    # Compute cosine similarity between feat and all targets
    # Remove from the row of each similarity where labels == assigned_targets_label
    # log(Exp().sum()).sum / feat.shape[0]

    # === Apply directly on features
    # feat = normalize(feat)
    # sim = F.cosine_similarity(feat[None,:,:], feat[:,None,:], dim=-1)
    # loss = torch.log(torch.exp(sim/1).sum(axis = 1)).sum() / feat.shape[0]

    # Average feat of the same class
    # And remove the the labels from the label
    # === Apply on averaged features
    # import pdb; pdb.set_trace()
    # unique_labels, inverse_indices, counts = torch.unique(labels, return_inverse=True, return_counts = True)
    # averaged = torch.zeros(len(unique_labels), feat.shape[1]).cuda()
    
    # for i, l in enumerate(unique_labels):
    #     label_indices = torch.where(labels == l)[0]
    #     averaged_row = torch.mean(feat[label_indices], dim=0)
    #     averaged[i] = averaged_row
    # averaged = normalize(averaged)

    # sim = F.cosine_similarity(averaged[None,:,:], averaged[:,None,:], dim=-1)
    # loss = torch.log(torch.exp(sim/1).sum(axis = 1)).sum() / averaged.shape[0]

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
    
    #====
    # targets = torch.cat((assigned_targets_not_in_batch, unassigned_targets))
    # pert_targets = perturb_targets_norm(targets, epsilon = 1e-2)
    # all_targets = normalize(torch.cat((averaged, pert_targets))) # turn off
    #====


    sim = F.cosine_similarity(all_targets[None,:,:], all_targets[:,None,:], dim=-1)
    loss = torch.log(torch.exp(sim/1).sum(axis = 1)).sum() / all_targets.shape[0]

    return loss
