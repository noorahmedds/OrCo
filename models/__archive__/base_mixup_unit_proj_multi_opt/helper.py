# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from wandb.plot import confusion_matrix
import torch.nn as nn
# from torch.autograd import Variable

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from .mixup import *

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

        # layer_mix = -1  # Refers to no mixup
        # if args.mixup_base: # input mixup
        #     layer_mix = 0
        # elif args.mixup_hidden: # select which layer to mixup at of the resnet
        #     layer_mix = random.choice(args.mixup_layer_choice) # Note: Selecting here or outside will not matter

        # if layer_mix >= 0:
        #     # If a layer is selected we choose to mix there. This may be the input image or an intermediate layer as well
        #     logits, targets_a, targets_b, lam = model.module.forward_mixup(data, targets = train_label, mixup = True, mixup_alpha = args.mixup_alpha, layer_mix = layer_mix)
        #     # logits = logits[:, :args.base_class]
        #     loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        # else:

        enc_out, proj_out = model(data)
        enc_out = enc_out[:, :args.base_class]
        proj_out = proj_out[:, :args.base_class]

        fc_loss = F.cross_entropy(enc_out, train_label, label_smoothing = args.label_smoothing)
        projection_loss = F.cross_entropy(proj_out, train_label)
        total_loss = fc_loss + projection_loss
        
        fc_acc = count_acc(enc_out, train_label)
        proj_acc = count_acc(proj_out, train_label)
        total_acc = (fc_acc + proj_acc) / 2.0

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} (fc:{:.3f}, proj:{:.3f}) acc={:.4f} (fc:{:.3f}, proj:{:.3f})'.format(epoch, lrc, total_loss.item(), fc_loss.item(), projection_loss.item(), total_acc, fc_acc, proj_acc))
        tl.add(total_loss.item())
        ta.add(total_acc)

        optimizer.zero_grad()
        total_loss.backward()
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
            
            if "multi" in model.module.mode:
                enc_out, proj_out = model(data)

                enc_out = enc_out[:, :test_class]
                proj_out = proj_out[:, :test_class]

                proj_loss = F.cross_entropy(proj_out, test_label)
                fc_loss = F.cross_entropy(enc_out, test_label)
                total_loss = fc_loss + proj_loss

                proj_acc = count_acc(proj_out, test_label)
                fc_acc = count_acc(enc_out, test_label)
                acc = (fc_acc + proj_acc) / 2.0
                
            else:
                enc_out = model(data)

                fc_loss = F.cross_entropy(enc_out, test_label)
                total_loss = fc_loss
                
                fc_acc = count_acc(enc_out, test_label)
                acc = fc_acc

            all_probs.append(enc_out)
            all_targets.append(test_label)

            # >>> Addition
            novelAcc, baseAcc = count_acc_(enc_out, test_label, test_class, args)

            if session > 1:
                # Caluclate the novel accuracy from the previous sessions
                prevNovelAcc = count_acc_previous(enc_out, test_label, test_class, args)
                vaPrevNovel.add(prevNovelAcc)

            # TODO: Check, if novelAcc is not None:
            vaNovel.add(novelAcc)
            vaBase.add(baseAcc)

            vl.add(total_loss.item())
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
