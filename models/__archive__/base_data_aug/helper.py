# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from wandb.plot import confusion_matrix
from torchvision import transforms

def create_augmented_pair(data):
    # For each data point create a pair of augmented images. 
    # Pass through the network and calculate the loss
    # transform = transforms.Compose([
    #             transforms.RandomResizedCrop(data.shape[-1], scale=(0.2, 1.)),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #             transforms.RandomGrayscale(p=0.2)])

    # SOURCE: https://github.com/CanPeng123/FSCIL_ALICE/blob/f976baf9ba0860756f4aadd5b29c285a423b9f8d/alice/dataloader/miniimagenet/miniimagenet.py

    # TODO: Update for all the rest of the datasets as well

    image_size = 84
    train_transforms = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2)
            ])
    
    return train_transforms(data), train_transforms(data)


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data1, data2, train_label = [_.cuda() for _ in batch]

        # TODO: Data augmentation. And average loss from the two data augmentations of the saem example
        # data1, data2 = create_augmented_pair(data)

        # Run data1 and data2 through the model. 
        logits = model(data1)[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label, label_smoothing = args.label_smoothing)

        logits = model(data2)[:, :args.base_class]
        loss += F.cross_entropy(logits, train_label, label_smoothing = args.label_smoothing)

        # Averaging the two losses
        loss = loss / 2.0

        # logits = model(data)
        # logits = logits[:, :args.base_class]
        # loss = F.cross_entropy(logits, train_label, label_smoothing = args.label_smoothing)
        acc = count_acc(logits, train_label)    # Accuracy is computed on the second augmentation for now

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


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainset.data_aug = False # Setting to false so as to not perform extra augmentation during prototype creation
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


def grid_search_test(model, testloader, epoch, args, session, base_sess = False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()

    # >>> Addition
    
    all_targets=[]
    all_probs=[]

    best_hm = None
    best_comb = None
    best_acc = None

    base_choices = np.linspace(0.005, 1, 5)
    novel_choices = np.linspace(0.2, 1, 5)
    combinations = list(itertools.product(base_choices, novel_choices))

    for c in combinations:
        base_t, novel_t = c

        vaBase = Averager() # Averager for novel classes only        
        vaNovel = Averager() # Averager for novel classes only
        va = Averager()

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                logits = model(data)
                logits = logits[:, :test_class]

                all_targets.append(test_label)
                all_probs.append(logits)

                loss = F.cross_entropy(logits, test_label)
                
                # Note with this method we are checking the novel classifier on all the labels
                # And the base on all the samples. 
                baseAcc, novelAcc, total_acc = count_acc_max(logits, test_label, session, args, base_t, novel_t)
                # novelAcc, baseAcc = count_acc_sub(logits, test_label, session, args, base_t, novel_t)

                # >>> Addition
                # novelAcc, baseAcc = count_acc_(logits, test_label, test_class, args)

                if novelAcc is not None:
                    vaNovel.add(novelAcc)

                if baseAcc is not None:
                    vaBase.add(baseAcc)

                vl.add(loss.item())
                va.add(total_acc.item())

            # >>> Addition 
            vaNovel = vaNovel.item()
            vaBase = vaBase.item()
            va = va.item()
        
        vhm = hm(vaNovel, vaBase)
        vam = am(vaNovel, vaBase)
    
        # if best_hm is None:
        #     best_hm = vhm
        #     best_acc = va
        #     best_comb = c
        # else:
        #     if vhm > best_hm:
        #         best_hm = vhm
        #         best_acc = va
        #         best_comb = c
        
        if best_acc is None:
            best_hm = vhm
            best_acc = va
            best_comb = c
        else:
            if va > best_acc:
                best_hm = vhm
                best_acc = va
                best_comb = c

        # print("Best_hm: ",best_hm, "HM: ", vhm, ", Base Temp: ", base_t, ", Novel Temp: ", novel_t, \
        #     "Novel Acc: ", vaNovel, "Base Acc: ", vaBase)
        print("Best_acc: ", best_acc, "acc: ", va, ", Base Temp: ", base_t, ", Novel Temp: ", novel_t, \
            "Novel Acc: ", vaNovel, "Base Acc: ", vaBase)
            
    return best_acc, best_hm, best_comb[0], best_comb[1]

def test(model, testloader, epoch, args, session, base_sess = False, max_inference=False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    # >>> Addition
    vaBase = Averager() # Averager for novel classes only        
    vaNovel = Averager() # Averager for novel classes only
    
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

        
    return vl, va, vaNovel, vaBase, vhm, vam, cm, cmNovel, cmSummary
