from utils import *
from tqdm import tqdm
import torch.nn.functional as F

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

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            logits, _ = model(data)
            logits = logits[:, :test_class]

            all_targets.append(test_label)
            all_probs.append(logits)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

    # Concatenate all_targets and probs
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs, axis=0)

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
    
    # Compute va using class-wise accuracy
    pred = torch.argmax(all_probs, dim=1)
    va = class_acc = (pred == all_targets).type(torch.cuda.FloatTensor).mean().item()
    
    return va, novel_cw.item(), base_cw.item()

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

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]

        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    prototypes = torch.stack(proto_list, dim=0)

    model.module.mode = og_mode
    
    return prototypes
