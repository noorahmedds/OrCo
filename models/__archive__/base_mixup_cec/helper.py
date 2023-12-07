# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from wandb.plot import confusion_matrix
import torch.nn as nn
from torch.autograd import Variable

from copy import deepcopy

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from .mixup import *

def replace_to_rotate(proto_tmp, query_tmp, args):
    for i in range(args.low_way):
        # random choose rotate degree
        rot_list = [90, 180, 270]
        sel_rot = random.choice(rot_list)
        if sel_rot == 90:  # rotate 90 degree
            # print('rotate 90 degree')
            proto_tmp[i::args.low_way] = proto_tmp[i::args.low_way].transpose(2, 3).flip(2)
            query_tmp[i::args.low_way] = query_tmp[i::args.low_way].transpose(2, 3).flip(2)
        elif sel_rot == 180:  # rotate 180 degree
            # print('rotate 180 degree')
            proto_tmp[i::args.low_way] = proto_tmp[i::args.low_way].flip(2).flip(3)
            query_tmp[i::args.low_way] = query_tmp[i::args.low_way].flip(2).flip(3)
        elif sel_rot == 270:  # rotate 270 degree
            # print('rotate 270 degree')
            proto_tmp[i::args.low_way] = proto_tmp[i::args.low_way].transpose(2, 3).flip(3)
            query_tmp[i::args.low_way] = query_tmp[i::args.low_way].transpose(2, 3).flip(3)
    return proto_tmp, query_tmp

def cec_base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()

    tqdm_gen = tqdm(trainloader)

    # Label generation for pseudo incremental learning phase
    label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
    label = label.type(torch.cuda.LongTensor) # Push to gpu

    for i, batch in enumerate(tqdm_gen, 1):
        data, true_label = [_.cuda() for _ in batch]
        
        k = args.episode_way * args.episode_shot    # Number of total samples per pseudo incremental task
        proto, query = data[:k], data[k:]           # Note for each batch we take the k elements as the support and the rest as query

        # Data is in [batch, height, width, channel]

        # sample low_way data for the support class
        proto_tmp = deepcopy(
            proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
            :args.low_shot,
            :args.low_way, :, :, :].flatten(0, 1))  # Reshaping such that first dim has shots, second controls the class
            # proto_tmp[0] would give all 0th shot for all the classes in the support class
            # We take only low_shots and low way classes and then reflatten in the first and second axis.
        query_tmp = deepcopy(   # Note that the query set should coincide with the support class but does not need the low_shot limitation
            query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,  
            :args.low_way, :, :, :].flatten(0, 1))

        # The candidates for the embedding prototypes are randomly rotated.
        proto_tmp, query_tmp = replace_to_rotate(proto_tmp, query_tmp, args)

        model.module.mode = 'encoder'
        data = model(data)

        # Get the latent vectors for the prototypes and query sets for the low way-low shot setting
        proto_tmp = model(proto_tmp)
        query_tmp = model(query_tmp)

        # Getting support and query embeddings without any rotation
        proto, query = data[:k], data[k:]

        # Reshaping the way-shot data for each access
        proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
        query = query.view(args.episode_query, args.episode_way, query.shape[-1])

        # Rotated Latent vector reshaped into [low_shot, low_way, channel]
        proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1]) 
        query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

        proto = proto.mean(0).unsqueeze(0)          # Prototypes generated for the actual way shot setting with the non rotated latent vectors
        proto_tmp = proto_tmp.mean(0).unsqueeze(0)  # Prototypes generated for the lway lshot setting with the latent z vectors

        # Concatenating the embedding with and wihtout rotations.
        proto = torch.cat([proto, proto_tmp], dim=1)   
        query = torch.cat([query, query_tmp], dim=1)

        # Add an extra dimension
        proto = proto.unsqueeze(0)
        query = query.unsqueeze(0)

        # Forward through our attention module where we apply cosine similarity. 
        # The aim being that the feature extractor and the attention module will get updated enough
        # that the prototypes generated will actually be stronger representations of the actual class means
        # In both the data space and the latent space
        logits = model.module._forward(proto, query)

        # Note the label term just contains a set form 0 to episode.way + low way which works because
        # Our data and embedding prototypes are concatenated and the cosine similarity then just assigned the query to a particular prototype each
        # In a series
        total_loss = F.cross_entropy(logits, label)

        acc = count_acc(logits, label)
        
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return tl, ta

def cec_test(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    # >>> Addition
    vaBase = Averager() # Averager for base classes only        
    vaNovel = Averager() # Averager for novel classes only
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]

            model.module.mode = 'encoder'
            
            # Simple latent space vector here. The method to test currently would be to create a classifier but we don't have one so far.
            query = model(data)
            query = query.unsqueeze(0).unsqueeze(0)

            proto = model.module.fc.weight[:test_class, :].detach()
            proto = proto.unsqueeze(0).unsqueeze(0)

            logits = model.module._forward(proto, query) # Here we get the best class

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            # >>> Addition
            novelAcc, baseAcc = count_acc_(logits, test_label, test_class, args)
            if novelAcc is not None:
                vaNovel.add(novelAcc)
            if baseAcc is not None:
                vaBase.add(baseAcc)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
        # >>> Addition 
        vaNovel = vaNovel.item()
        vaBase = vaBase.item()

    vhm = hm(vaNovel, vaBase)

    return vl, va, vhm

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # import pdb; pdb.set_trace()
    criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        data, targets_a, targets_b, lam = mixup_data(data, train_label, alpha = args.mixup_alpha)
        data, targets_a, targets_b = map(Variable, (data, targets_a, targets_b))

        logits = model(data)

        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model

def test(model, testloader, epoch, args, session, base_sess = False, max_inference=False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    # >>> Addition
    vaBase = Averager() # Averager for novel classes only        
    vaNovel = Averager() # Averager for novel classes only
    vaPrevNovel = Averager()
    
    all_targets=[]
    all_probs=[]

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            logits = model(data)
            logits = logits[:, :test_class]

            all_targets.append(test_label)
            all_probs.append(logits)

            loss = F.cross_entropy(logits, test_label)
            
            if max_inference:
                acc = count_acc_max(logits, test_label, session, args)
            else:
                acc = count_acc(logits, test_label)

            # >>> Addition
            novelAcc, baseAcc = count_acc_(logits, test_label, test_class, args)

            # TODO: Caluclate the novel accuracy from the previous sessions
            if session > 1:
                prevNovelAcc = count_acc_previous(logits, test_label, test_class, args)
                vaPrevNovel.add(prevNovelAcc)

            # TODO: Check, if novelAcc is not None:
            vaNovel.add(novelAcc)
            vaBase.add(baseAcc)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

        # >>> Addition 
        vaNovel = vaNovel.item()
        vaBase = vaBase.item()
        vaPrevNovel = vaPrevNovel.item()

    vhm = hm(vaNovel, vaBase)
    vam = am(vaNovel, vaBase)

    all_targets = torch.cat(all_targets, axis = 0)
    cm = createConfusionMatrix( #plot_confusion_matrix(
        all_targets.cpu().numpy(), 
        torch.argmax(torch.cat(all_probs, axis = 0), axis = 1).cpu().numpy(),
        [str(i) for i in range(test_class)],
        hline_at = args.base_class,
        vline_at = args.base_class,
        session = session
    )

    cmSummary = None
    cmNovel = None
    if not base_sess:
        cmSummary = createConfusionMatrix( #plot_confusion_matrix(
            all_targets.cpu().numpy(), 
            torch.argmax(torch.cat(all_probs, axis = 0), axis = 1).cpu().numpy(),
            [str(i) for i in range(test_class)],
            hline_at = args.base_class,
            vline_at = args.base_class,
            summarize = True,
            session = session
        )

        # Creating confusion matrix just for the novel classes on the novel sub space
        novel_idx = all_targets >= args.base_class
        novel_probs = torch.cat(all_probs, axis = 0)[novel_idx, args.base_class:]
        novel_targets = all_targets[novel_idx] - args.base_class
        cmNovel = createConfusionMatrix( #plot_confusion_matrix(
            novel_targets.cpu().numpy(), 
            torch.argmax(novel_probs, axis = 1).cpu().numpy(),
            [str(i) for i in range(args.base_class, test_class)],
            cmap="crest",
            session = session
        )

        
    return vl, va, vaNovel, vaBase, vhm, vam, cm, cmNovel, cmSummary, vaPrevNovel
