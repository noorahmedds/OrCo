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
from .supcon import *

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target):
        probs = F.softmax(logits, 1) 
        loss = (- target * torch.log(probs)).sum(1).mean()
        return loss

def _mix_step(x0, z0, z1, mix_temperature = 0.05):
    crit = SoftCrossEntropy()
    mix_data = []
    mix_projections = []
    mix_lams = []

    k_index = torch.randperm(x0.shape[0])
    bsz = x0.shape[0]
    for i in range(bsz):
        # if labels[i] != labels[k_index][i]:
        lam = np.random.uniform(0.0, 1.0)
        mix_lams.append(lam)
        mix_data.append(lam * x0[i] + (1 - lam) * x0[k_index, :][i])
        mix_projections.append(lam * z0[i] + (1 - lam) * z0[k_index, :][i])

    mix_data = torch.stack(mix_data)
    mix_projections = torch.stack(mix_projections)
    mix_projections = F.normalize(mix_projections, dim=1)
    mix_lams = torch.tensor(mix_lams)
    lbls_mix = torch.cat((torch.diag(mix_lams), torch.diag(1-mix_lams)), dim=1).cuda()

    z = torch.cat((z1, z1[k_index]), axis = 0)
    logits_mix = torch.mm(mix_projections, z.transpose(0,1)) 
    logits_mix /= mix_temperature
    loss = crit(logits_mix, lbls_mix) / 2
        
    return loss

def sup_con_pretrain(model, criterion, trainloader, optimizer, scheduler, epoch, args, mixup_weigth = 1):
    tl = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        images, labels = batch

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # only for large batch size
        warmup_learning_rate(args, epoch, i, len(trainloader), optimizer)

        # compute loss
        features = model.module.forward_sup_con(images)
        # features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        sc_loss = criterion(features, labels, margin = args.margin_sup_con)

        lrc = optimizer.param_groups[0]["lr"]
        if args.mixup_sup_con:
            x0 = images[:bsz, ...]
            mixup_loss = _mix_step(x0, f1, f2)
            total_loss = sc_loss + (mixup_weigth*mixup_loss)
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f}, sc_loss {:.3f}, mixup_loss {:.3f}'.format(epoch, lrc, sc_loss.item(), mixup_loss.item()))
        else:
            total_loss = sc_loss
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f}, sc_loss {:.3f}'.format(epoch, lrc, sc_loss.item()))

        # lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    return tl

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # import pdb; pdb.set_trace()
    criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
    supcon_criterion = SupConLoss()
    # standard classification for pretrain

    sup_con_toggle = True
    if args.two_step:
        sup_con_toggle = False

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

        # Sup Con Loss
        sc_loss = 0
        if args.sup_con_base and sup_con_toggle:
            if args.use_sup_con_head:
                encoding = model.module.sup_con_head(encoding)  
            sc_loss = supcon_criterion(encoding.unsqueeze(1), train_label)

        acc = count_acc(logits, train_label)

        total_loss = loss + sc_loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}, (cls_loss, sc_loss) ({:.3f}, {:.3f}) acc={:.4f}'.format(epoch, lrc, total_loss.item(), loss.item(), sc_loss.item() if args.sup_con_base and sup_con_toggle else 0, acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if args.two_step: sup_con_toggle = not sup_con_toggle   # toggle sup con for next batch

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
            
            logits, encoding = model(data)
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


def get_base_fc(trainset, transform, model, args):
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
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            
            # OLD TODO: Do normalisation here. I imagine this will improve results
            # embedding = F.normalize(model(data), dim = 1)
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

    # Get reserved prototypes i.e. for every neighbouring prototype create a mid point vector
    # Here we get 200 such reserved vectors which are equidistant from each other and the base protoypes
    prototypes = torch.stack(proto_list, dim=0)

    # Normalise the prototypes. THey already get normalised in the cosine classifier but i think for reserve generation could be important
    if not args.skip_encode_norm:
        prototypes = F.normalize(prototypes, dim=1)

    model.module.mode = og_mode

    return prototypes