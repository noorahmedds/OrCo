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
from .dino_utils import *

def s2m2_train(model, trainloader, optimizer, scheduler, epoch, args, lr_schedule, wd_schedule, momentum_schedule):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # import pdb; pdb.set_trace()
    criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    # standard classification for pretrain

    do_mixup = epoch >= args.mixup_start_epoch 

    losses = {}

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        it = (len(trainloader) * epoch) + i-1 # global training iteration

        # Scheduling lr
        if args.schedule_lr:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                
                if i == 0 and args.schedule_wd:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

        data, train_label, all_crops = batch
        data = data.cuda()
        train_label = train_label.cuda()
        bs = data.shape[0]

        optimizer.zero_grad()

        if do_mixup:
            layer_mix = -1  # Refers to no mixup
            if args.mixup_base: # input mixup
                layer_mix = 0
            elif args.mixup_hidden: # select which layer to mixup at of the resnet
                layer_mix = random.choice(args.mixup_layer_choice) # Note: Selecting here or outside will not matter

            if layer_mix >= 0:
                logits, targets_a, targets_b, lam = model.module.forward_mixup(data, targets = train_label, mixup = True, mixup_alpha = args.mixup_alpha, layer_mix = layer_mix)
                mixup_loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                logits = model(data)
                mixup_loss = F.cross_entropy(logits, train_label, label_smoothing = args.label_smoothing)
            # mixup_loss.backward(retain_graph = True)
            losses["mixup_loss"] = mixup_loss

        # For each data point do the augmentation
        # Traverse all samples in the first dim

        dino_loss = 0
        if args.all_crops:
            ixs = list(range(bs))
        else:
            ixs = [random.randint(0,bs-1)]   
        cls_batch = []
        for j in ixs:  # Selected batches
            crops = []
            cls_batch.extend([all_crops[0][j], all_crops[1][j]]) # appending the global views
            for v in all_crops:
                crops.append(v[j]) # crop (v) for the selected batch j
            student_output, teacher_output = model.module.forward_dino(crops, epoch)
            dino_loss += compute_dino_loss(student_output, teacher_output)
            del student_output, teacher_output
        dino_loss = dino_loss / len(ixs)
        losses["dino_loss"] = dino_loss * args.dino_loss_weight
        # dino_loss *= 0.1

        # convert cls_batch to tensor
        # make target tensor with repeat interleave
        # permute cls_batch and apply same to target
        # pass through model
        # run through label smoothing criterion

        cls_batch = torch.stack(cls_batch)
        cls_label = train_label.repeat_interleave(2)
        rand_ixs = torch.randperm(cls_batch.shape[0])
        cls_batch = cls_batch[rand_ixs]
        cls_label = cls_label[rand_ixs]
        logits = model(cls_batch)
        cls_loss = F.cross_entropy(logits, cls_label, label_smoothing = args.label_smoothing)
        losses["cls_loss"] = cls_loss

        # logits = model(data)
        # cls_loss = F.cross_entropy(logits, train_label)
        # losses["cls_loss"] = cls_loss

        total_loss = 0
        loss_string = ""
        for k, v in losses.items():
            total_loss += v
            loss_string += "{}: {:.3f} | ".format(k, v.item())
        loss_string += f" total_loss = {total_loss.item():.3f}"

        acc = count_acc(logits, cls_label)
        # acc = count_acc(logits, train_label)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f}, acc={:.4f},{}'.format(epoch,lrc,acc,loss_string))
        tl.add(total_loss.item())
        ta.add(acc)

        total_loss.backward()
        # loss.backward(retain_graph = True)
        # dino_loss.backward()
        
        # Cancel gradients for the first epoch
        cancel_gradients_last_layer(epoch, model.module.student, 1)
        
        optimizer.step()
        
        # Update the teacher with moving average
        # model.module.update_teacher()
        model.module.update_teacher(momentum_teacher = momentum_schedule[it])

    tl = tl.item()
    ta = ta.item()
    return tl, ta

def base_train(model, trainloader, optimizer, scheduler, epoch, args, lr_schedule, wd_schedule, momentum_schedule):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # import pdb; pdb.set_trace()
    criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    # standard classification for pretrain

    dino_toggle = True # if true then dino loss is computed
    if args.two_step:
        dino_toggle = False # if true then dino loss is computed

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        it = (len(trainloader) * epoch) + i-1 # global training iteration

        # Scheduling lr
        if args.schedule_lr:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                
                if i == 0 and args.schedule_wd:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

        data, train_label, all_crops = batch
        data = data.cuda()
        train_label = train_label.cuda()
        bs = data.shape[0]

        layer_mix = -1  # Refers to no mixup
        if args.mixup_base: # input mixup
            layer_mix = 0
        elif args.mixup_hidden: # select which layer to mixup at of the resnet
            layer_mix = random.choice(args.mixup_layer_choice) # Note: Selecting here or outside will not matter

        if layer_mix >= 0:
            logits, targets_a, targets_b, lam = model.module.forward_mixup(data, targets = train_label, mixup = True, mixup_alpha = args.mixup_alpha, layer_mix = layer_mix)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)    # Note this is mixup with label smoothing so the criterion is smoothening the 2 output labels. The combination of these two parameters is quite crucial to the dataset. Right now it is 0.3 and 0.95 (i.e. 1 - 0.95 + small alpha on the rest of the nodes) 
        else:
            logits = model(data)
            loss = F.cross_entropy(logits, train_label, label_smoothing = args.label_smoothing)

        # For each data point do the augmentation
        # Traverse all samples in the first dim
        dino_loss = 0
        if dino_toggle:
            if args.class_level:
                # find a random class that has more than 1 sample in this batch
                # if found then arbitrarily combine the crops for both. 
                unique_label, counts = torch.unique(train_label, return_counts = True)
                filtered_counts = torch.argwhere(counts > 1).flatten()
                if filtered_counts.shape[0] < 2:
                    ix = random.randint(0,bs-1)
                    crops = []
                    for v in all_crops:
                        crops.append(v[ix]) # crop (v) for the selected batch j
                else:        
                    temp = random.randint(0, filtered_counts.shape[0]-1)
                    selected_label = unique_label[filtered_counts[temp]]

                    # for this selected label get all batch sample idx
                    class_sample_ixs = torch.argwhere(train_label == selected_label).flatten()
                    
                    # choose 2 of the above sample
                    class_sample_ixs = class_sample_ixs[torch.randperm(class_sample_ixs.size(0))[:2]]

                    crops = []
                    for v in all_crops: crops.append(v[random.choice(class_sample_ixs)]) 

                student_output, teacher_output = model.module.forward_dino(crops, epoch)
                dino_loss = compute_dino_loss(student_output, teacher_output)
                del student_output, teacher_output
            else:
                if args.all_crops:
                    ixs = list(range(bs))
                else:
                    ixs = [random.randint(0,bs-1)]   
                for j in ixs:  # Selected batches
                    crops = []
                    for v in all_crops:
                        crops.append(v[j]) # crop (v) for the selected batch j
                    student_output, teacher_output = model.module.forward_dino(crops, epoch)
                    dino_loss += compute_dino_loss(student_output, teacher_output)
                    del student_output, teacher_output
                dino_loss = dino_loss / len(ixs)

                # Toggle for old
                # j = random.randint(0,bs-1)
                # crops = []
                # for v in all_crops:
                #     crops.append(v[j])
                # student_output, teacher_output = model.module.forward_dino(crops)
                # dino_loss += compute_dino_loss(student_output, teacher_output)
                # # The intermediat outputs need to be deleted because they live past the loop
                # # as found here: https://pytorch.org/docs/stable/notes/faq.html
                # del student_output, teacher_output
                # dino_loss = dino_loss # / bs

            dino_loss *= args.dino_loss_weight
        total_loss = loss + dino_loss
        
        acc = count_acc(logits, train_label)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f}, total loss={:.4f} (cls, dino) = ({:.4f}, {:.4f}), acc={:.4f}'.format(
                    epoch, 
                    lrc, 
                    total_loss.item(), 
                    loss.item(), 
                    dino_loss,#.item(), 
                    acc))
        tl.add(total_loss.item())
        ta.add(acc)

        # TODO: Backpass first on the cls loss with opt1
        # TODO: Then backpass with dino loss with opt1

        optimizer.zero_grad()
        # total_loss.backward()
        loss.backward(retain_graph = True)
        if dino_toggle:
            dino_loss.backward()
        
        # Cancel gradients for the first epoch
        cancel_gradients_last_layer(epoch, model.module.student, 1)
        
        optimizer.step()
        
        # Update the teacher with moving average
        try:
            model.module.update_teacher(momentum_teacher = momentum_schedule[it])
        except Exception as e:
            import pdb; pdb.set_trace()

        # if args.two_step:
        #     dino_toggle = not dino_toggle

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

def set_base_avg(trainset, transform, model, args):
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label, crops = batch
            data = data.cuda()
            label = label.cuda()
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding)
    embedding_list = torch.cat(embedding_list, dim=0)
    model.module.base_avg_encoding = embedding_list.mean(0)

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
            data, label, crops = batch
            data = data.cuda()
            label = label.cuda()
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

    if args.classifier_last_layer == "projection":
        model.module.fc[-1].weight.data[:args.base_class] = proto_list
    
        # Remove the projection part of the fc layer
        model.module.fc = model.module.fc[-1]
    elif args.classifier_last_layer == "linear":
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

            if session > 1:
                # Caluclate the novel accuracy from the previous sessions
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
