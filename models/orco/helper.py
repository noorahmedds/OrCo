# import new Network name here and add in model_class args
from Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from wandb.plot import confusion_matrix
import torch.nn as nn
# from torch.autograd import Variable


import os

from mixup import *
from supcon import *

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    cw_acc = Averager()
    base_cw = Averager()
    novel_cw = Averager()
    
    all_targets=[]
    all_probs=[]
    all_projections = []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            logits, projection = model(data)
            logits = logits[:, :test_class]

            all_targets.append(test_label)
            all_probs.append(logits)
            all_projections.append(projection)

            loss = F.cross_entropy(logits, test_label)
            
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

    # Concatenate all_targets and probs
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs, axis=0)

    # Compute false positives
    fpr = {
            "novel2base": 0,    # Count of samples from novel classes selected as base / Total novel samples
            "base2novel": 0,    # Count of samples from base classes selected as novel 
            "base2base": 0,     # Count of samples from base classes selected as other base
            "novel2novel": 0,    # Count of samples from novel classes selected as other novels
            "total_novel": 0,
            "total_base": 0
        }
    fpr = count_fp(all_probs, all_targets, test_class, args, fpr)
    fpr["base2base"] /= fpr["total_base"]
    if session > 0:
        fpr["novel2base"] /= fpr["total_novel"]
        fpr["base2novel"] /= fpr["total_base"]
        fpr["novel2novel"] /= fpr["total_novel"]

    # Now compute vaNovel as
    novel_mask = all_targets >= args.base_class
    pred = torch.argmax(all_probs, dim=1)[novel_mask]
    label_ = all_targets[novel_mask]
    novel_acc = (pred == label_).type(torch.cuda.FloatTensor).mean().item()

    # Compute base acc as
    base_mask = all_targets < args.base_class
    pred = torch.argmax(all_probs, dim=1)[base_mask]
    label_ = all_targets[base_mask]
    base_acc = (pred == label_).type(torch.cuda.FloatTensor).mean().item()

    vaSession = count_acc_session(all_probs, all_targets, args)

    # Compute class wise accuracy
    for l in all_targets.unique():
        # Get class l mask
        class_mask = all_targets == l
        pred = torch.argmax(all_probs, dim=1)[class_mask]
        label_ = all_targets[class_mask]
        class_acc = (pred == label_).type(torch.cuda.FloatTensor).mean().item()
        cw_acc.add(class_acc)
        
        if l < args.base_class:
            base_cw.add(class_acc)
        else:
            novel_cw.add(class_acc)
        
    cw_acc = cw_acc.item()
    
    # Compute va using class-wise accuracy
    pred = torch.argmax(all_probs, dim=1)
    va = class_acc = (pred == all_targets).type(torch.cuda.FloatTensor).mean().item()
    
    cos_sims = {
        "inter":0,
        "intra":0
    }

    cos_sims["inter"] = compute_inter_class_cosine_similarity(projection_features, all_targets)
    cos_sims["intra"] = compute_intra_class_cosine_similarity(projection_features, all_targets)
    
    return vl, va, novel_acc, base_acc, vaSession, cw_acc, novel_cw.item(), base_cw.item(), cos_sims, fpr

def get_base_prototypes(trainset, transform, model, args, mode="encoder"):
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    og_mode = model.module.mode
    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        tqdm_gen.set_description("Generating Features: ")
        model.module.mode = mode
        for i, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            
            embedding = model(data)

            embedding_list.append(embedding)
            label_list.append(label)

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    cov_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]

        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    prototypes = torch.stack(proto_list, dim=0)

    model.module.mode = og_mode
    
    return prototypes

def compute_intra_class_cosine_similarity(projection_features, test_targets):
    test_projection_features = normalize(projection_features)

    avg_cosine_sim = Averager()
    for i in test_targets.unique():
        mask = torch.argwhere(test_targets==i).flatten()
        curr_features = test_projection_features[mask]
        cosine_sim = torch.matmul(curr_features, curr_features.T)
        logits_mask = torch.scatter(torch.ones_like(cosine_sim), 1, torch.arange(curr_features.shape[0]).view(-1, 1).to(curr_features.device),0)
        avg_cosine_sim.add((logits_mask*cosine_sim).mean())
    
    return avg_cosine_sim.item()

def compute_inter_class_cosine_similarity(projection_features, test_targets):
    test_projection_features = normalize(projection_features)

    avg_cosine_sim = Averager()
    for i in test_targets.unique():
        mask = torch.argwhere(test_targets==i).flatten()
        curr_features = test_projection_features[test_targets==i]
        curr_not_features = test_projection_features[test_targets!=i]
        cosine_sim = torch.matmul(curr_features, curr_not_features.T)
        avg_cosine_sim.add(cosine_sim.mean())
    
    return avg_cosine_sim.item()