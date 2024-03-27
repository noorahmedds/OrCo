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
# from knn_classifier import WeightedKNNClassifier

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    cw_acc = Averager()
    base_cw = Averager()
    novel_cw = Averager()

    # vaBase = Averager() # Averager for novel classes only        
    # vaNovel = Averager() # Averager for novel classes only
    # vaPrevNovel = Averager()

    # model.eval()   # Turn off
    
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
    # Compute inter and intra class cosine similarity
    projection_features = torch.cat(all_projections, axis = 0)
    if 0 not in model.module.fc.rv.shape:
        cos_sim_to_unassigned = F.cosine_similarity(projection_features[None,:,:], model.module.fc.rv[:,None,:], dim=-1).mean()
        print(f"===> For session {session}: The cosine similarity for all projected test features to unassigned targets = {cos_sim_to_unassigned:.3f}")

    cos_sims["inter"] = compute_inter_class_cosine_similarity(projection_features, all_targets)
    cos_sims["intra"] = compute_intra_class_cosine_similarity(projection_features, all_targets)
    
    return vl, va, novel_acc, base_acc, vaSession, cw_acc, novel_cw.item(), base_cw.item(), cos_sims, fpr

def sup_con_pretrain(model, criterion, trainloader, testloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    model.module.mode = "sup_con" 
    if args.mixup_sup_con: model.module.mode += "_mixup"
    for i, batch in enumerate(tqdm_gen, 1):
        images, labels = batch
    
        # if args.mixup_sup_con:
        #     # i0, i1, labels = fusion_aug_two_image(images[0].cuda(), images[1].cuda(), labels.cuda(), 0, args, mix_times = 1)
        #     # images = torch.cat([i0, i1], dim=0)
        # else:
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        bsz = labels.shape[0]

        # only for large batch size
        warmup_learning_rate(args, epoch, i, len(trainloader), optimizer)

        # compute loss
        # features = model.module.forward_sup_con(images)
        if not args.mixup_sup_con:
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sc_loss = criterion(features, labels, margin = args.margin_sup_con)
        else:
            features, targets_a, targets_b, lam, encoding = model(images, targets = labels, mixup = True, mixup_alpha = args.mixup_alpha, layer_mix = 0, sup_con = True)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sc_loss = mixup_criterion(criterion, features, targets_a, targets_b, lam)
            
        # encoding = model(data)
        # if args.use_sup_con_head:
        #     encoding = model.module.sup_con_head(encoding)  
        # logits = model.module.forward_sup_con(data)
        # sc_loss = supcon_criterion(logits.unsqueeze(1), train_label)
        # acc = count_acc(logits, train_label)

        total_loss = sc_loss

        # lrc = scheduler.get_last_lr()[0]
        lrc = optimizer.param_groups[0]["lr"]
        # tqdm_gen.set_description(
        #     'Session 0, epo {}, lrc={:.4f}, sc_loss {:.3f}, selement_loss {:.3f}'.format(epoch, lrc, sc_loss.item()))
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f}, sc_loss {:.3f}'.format(epoch, lrc, sc_loss.item()))
        tl.add(total_loss.item())

        optimizer.zero_grad()
        sc_loss.backward()
        optimizer.step()

    # Test the model here with the test loader to see the validation loss on the test set
    tqdm_gen = tqdm(trainloader)

    model.module.mode = args.base_mode
    tl = tl.item()
    return tl

def ce_pretrain(model, criterion, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    model.module.mode = "cross_entropy"
    for i, batch in enumerate(tqdm_gen, 1):
        images, labels = batch
            
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute loss
        logits = model(images)
        ce_loss = criterion(logits, labels)

        total_loss = ce_loss

        lrc = optimizer.param_groups[0]["lr"]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f}, ce_loss {:.3f}'.format(epoch, lrc, ce_loss.item()))
        tl.add(total_loss.item())

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

    model.module.mode = args.base_mode
    tl = tl.item()
    return tl

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # import pdb; pdb.set_trace()
    criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    # supcon_criterion = SupConLoss()
    # standard classification for pretrain

    # sup_con_toggle = True
    # if args.two_step:
    #     sup_con_toggle = False

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        layer_mix = -1  # Refers to no mixup
        if args.mixup_base: # input mixup
            layer_mix = 0
        elif args.mixup_hidden: # select which layer to mixup at of the resnet
            layer_mix = random.choice(args.mixup_layer_choice) # Note: Selecting here or outside will not matter

        if layer_mix >= 0:
            logits, targets_a, targets_b, lam, encoding = model.module.forward_mixup(data, targets = train_label, mixup = True, mixup_alpha = args.mixup_alpha, layer_mix = layer_mix)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        else:
            logits, encoding = model(data)
            loss = F.cross_entropy(logits, train_label, label_smoothing = args.label_smoothing)


        # # Sup Con Loss
        sc_loss = 0
        # if args.sup_con_base and sup_con_toggle:
        #     if args.use_sup_con_head:
        #         encoding = model.module.sup_con_head(encoding)  
        #     sc_loss = supcon_criterion(encoding.unsqueeze(1), train_label)

        acc = count_acc(logits, train_label)

        total_loss = loss + sc_loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, (cls_loss, sc_loss) ({:.3f}, {:.3f}) acc={:.4f}'.format(epoch, lrc, total_loss.item(), loss.item(), 0, acc)) # sc_loss.item() if args.sup_con_base and sup_con_toggle else 
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # if args.two_step: sup_con_toggle = not sup_con_toggle   # toggle sup con for next batch

    tl = tl.item()
    ta = ta.item()
    return tl, ta

import collections

def replace_mixup_fc(model, args):
    if type(model) == collections.OrderedDict:
        mixup_fc_tensor = model["module.fc.weight"]
        fc = nn.Linear(mixup_fc_tensor.shape[1], args.num_classes, bias=False).cuda()       # Note the entire number of classes are already added
        
        fc.weight.data[:args.base_class] = mixup_fc_tensor[:args.base_class]

        model["module.fc.weight"] = fc.weight
    else:
        fc = nn.Linear(model.module.num_features, args.num_classes, bias=False).cuda()       # Note the entire number of classes are already added
        mixup_fc = model.module.fc
        fc.weight.data[:args.base_class] = mixup_fc.weight.data[:args.base_class]

        # Replace the model fc after the base session with the original num of classes length
        model.module.fc = fc

    return model

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    # if args.classifier_last_layer == "projection":
    #     model.module.fc[-1].weight.data[:args.base_class] = proto_list
    
    #     # Remove the projection part of the fc layer
    #     model.module.fc = model.module.fc[-1]
    # else:
    #     model.module.fc.weight.data[:args.base_class] = proto_list

    model.module.fc.classifiers[0].weight.data = proto_list

    return model

def replace_base_fc_balanced(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []    

    # Here we have a whole data protolist. 
    # For each class now we find K classes which are closest to its average prototype
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_mean = embedding_this.mean(0)

        # Calculate cosine similarity
        embedding_this_norm = embedding_this / embedding_this.norm(dim=1)[:, None]
        embedding_mean_norm = embedding_mean / embedding_mean.norm()
        res = torch.matmul(embedding_this_norm, embedding_mean_norm)

        # Now we find top K indices in res that correspond to the 
        top_k_ix = res.topk(args.shot).indices

        # Appending the mean to the protolist
        proto_list.append(embedding_this[top_k_ix].mean(0))

    proto_list = torch.stack(proto_list, dim=0)
    model.module.fc.weight.data[:args.base_class] = proto_list

    return model

def get_base_fc(trainset, transform, model, args, return_embeddings=False, return_cov=False, mode="encoder"):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    og_mode = model.module.mode
    # data_list=[]
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
        
        cov_this = np.cov(normalize(embedding_this).cpu(), rowvar=False)
        cov_list.append(cov_this)

        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    # Get reserved prototypes i.e. for every neighbouring prototype create a mid point vector
    # Here we get 200 such reserved vectors which are equidistant from each other and the base protoypes
    # Skip Normalise the prototypes to ensure that. Normlalisation happens only in the classifier class
    prototypes = torch.stack(proto_list, dim=0)

    model.module.mode = og_mode

    if return_embeddings:
        prototypes = (prototypes, embedding_list, label_list)

    if return_cov:
        prototypes = (prototypes, cov_list)
    
    return prototypes

# from geom_median.torch import compute_geometric_median
    
def get_base_gm(trainset, transform, model, args, return_embeddings=False, mode="encoder"):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    og_mode = model.module.mode
    # data_list=[]
    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        tqdm_gen.set_description("Generating Features: ")
        model.module.mode = mode
        for i, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            
            # OLD TODO: Do normalisation here. I imagine this will improve results
            # embedding = F.normalize(model(data), dim = 1)
            embedding = model(data)

            embedding_list.append(embedding)
            label_list.append(label)
    
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]

        # compute geometric median
        embedding_this = compute_geometric_median(embedding_this.cpu()).median.cuda()
        # embedding_this = compute_geometric_mean(embedding_this)

        proto_list.append(embedding_this)

    # Get reserved prototypes i.e. for every neighbouring prototype create a mid point vector
    # Here we get 200 such reserved vectors which are equidistant from each other and the base protoypes
    prototypes = torch.stack(proto_list, dim=0)

    # Because we are no looking at the geometric median this is not a problem
    if not args.skip_encode_norm:
        prototypes = F.normalize(prototypes, dim=1)

    model.module.mode = og_mode

    if return_embeddings:
        prototypes = (prototypes, embedding_list, label_list)
        
    return prototypes

def sup_con_test(model, criterion, testloader, optimizer, scheduler, epoch, args):
    vl = Averager()
    va = Averager()
    model = model.eval()
    tqdm_gen = tqdm(testloader)
    model.module.mode = "sup_con"
    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen, 1):
            images, labels = batch

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # compute loss
            # features = model.module.forward_sup_con(images)
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sc_loss = criterion(features, labels, margin = args.margin_sup_con)

            total_loss = sc_loss

            # TODO: Get encodings from the model
            # encoding = self.encode(data).detach()
            # logits = self.get_logits(encoding, fc)
            # va.add(count_acc(logits, labels))

            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f}, sc_loss {:.3f}'.format(epoch, lrc, sc_loss.item()))
            vl.add(total_loss.item())

    model.module.mode = args.base_mode
    vl = vl.item()
    va = va.item()
    return vl, va
    

def get_new_fc(trainset, transform, model, args, class_list, views=1, return_embeddings=False, mode="encoder"):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    og_mode = model.module.mode
    # data_list=[]
    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        tqdm_gen.set_description("Generating Features: ")
        model.module.mode = mode
        for view in range(views):
            for i, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                
                embedding = model(data)

                embedding_list.append(embedding)
                label_list.append(label)
    
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in class_list:
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    # Get reserved prototypes i.e. for every neighbouring prototype create a mid point vector
    # Here we get 200 such reserved vectors which are equidistant from each other and the base protoypes
    # Skip Normalise the prototypes to ensure that. Normlalisation happens only in the classifier class
    prototypes = torch.stack(proto_list, dim=0)

    model.module.mode = og_mode

    if return_embeddings:
        prototypes = (prototypes, embedding_list, label_list)
        
    return prototypes


def extract_features(model, loader):
    features = []
    labels = []
    with torch.no_grad():
        tqdm_gen = tqdm(loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            
            _, feat = model(data)
            features.append(feat.detach())
            labels.append(label)

    features = torch.cat(features, axis = 0)
    # features = F.normalize(features, p=2, dim=-1)

    labels = torch.cat(labels)
    return features, labels

def test_knn(model, trainset, testloader):
    model = model.eval()
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = testloader.dataset.transform
    
    train_features, train_targets = extract_features(model, trainloader)
    test_features, test_targets = extract_features(model, testloader)
    
    knn = WeightedKNNClassifier(
        k=5,
        distance_fx="cosine",
    )
    
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )
    
    acc1, acc5 = knn.compute()
    
    del knn
    
    return acc1, acc5

def compute_intra_class_cosine_similarity(projection_features, test_targets):
    test_projection_features = normalize(projection_features)

    avg_cosine_sim = Averager()
    for i in test_targets.unique():
        # Now filter the features for this target
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
        # Now filter the features for this target
        mask = torch.argwhere(test_targets==i).flatten()
        curr_features = test_projection_features[test_targets==i]
        curr_not_features = test_projection_features[test_targets!=i]
        cosine_sim = torch.matmul(curr_features, curr_not_features.T)
        avg_cosine_sim.add(cosine_sim.mean())
    
    return avg_cosine_sim.item()