import numpy as np
import torch

def mixup_pairs(batch, targets, curr_mix_pair, base_class, alpha=20.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        if lam < 0.4 or lam > 0.6:
            lam = 0.5
        # lam = 0.5
    else:
        lam = 1

    bsz = targets.shape[0]

    batch1, batch2 = torch.split(batch, [bsz, bsz], dim=0)

    mixed_batch1 = []
    mixed_batch2 = []
    mixed_labels = []
    for ix, y in enumerate(targets):
        # Given y. Find location of y in curr_mix_pair, 
        # the location detrmines the mixing pair
        pair = torch.argwhere(curr_mix_pair == y).flatten()
        # pos is the target label sample this ix needs to be paired with
        # find this pos in the batch.
        if pair.numel() > 0:
            pos = torch.argwhere(targets == pair).flatten()
            # select pos randomly
            if pos.numel() > 0:
                if pos.numel() > 1:
                    sel_ind = torch.randperm(pos.numel())[0]
                    pos = pos[sel_ind].unsqueeze(0)
                mixed_batch1.append(lam * batch1[ix] + (1 - lam) * batch2[pos])
                mixed_batch2.append(lam * batch2[ix] + (1 - lam) * batch1[pos])
                mixed_labels.append(pair + base_class)
    
    if len(mixed_batch1) > 0:
        mixed_batch1 = torch.cat(mixed_batch1)
        mixed_batch2 = torch.cat(mixed_batch2)
        mixed_labels = torch.cat(mixed_labels)
        mixed_batch = torch.cat([batch1, mixed_batch1, batch2, mixed_batch2], dim = 0)
        mixed_labels = torch.cat([targets, mixed_labels])

        return mixed_batch, mixed_labels
    else:
        return batch, targets

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def instance_mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Mixup only data of the same class
    batch_size = x.size()[0]

    output = []
    targets = []
    for i in y.unique():
        ix = torch.argwhere(y == i).flatten()
        ix_hat = ix[torch.randperm(ix.shape[0])].flatten()
        output.append((lam*x[ix] + (1-lam)*x[ix_hat]))  # Instance mixing to generate 
        targets.append(y[ix])

    mixed_x = torch.cat(output)
    mixed_targets = torch.cat(targets)
    return mixed_x, mixed_targets

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def fusion_aug_two_image(x_1, x_2, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x_1.size()[0]
    mix_data_1 = []
    mix_data_2 = []
    mix_target = []

    # print('fusion_aug_two_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data_1.append(lam * x_1[i] + (1 - lam) * x_1[index, :][i])  # Each augmentation will now gets its own mixup but only with the same class
                mix_data_2.append(lam * x_2[i] + (1 - lam) * x_2[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data_1:
        x_1 = torch.cat((x_1, item.unsqueeze(0)), 0)
    for item in mix_data_2:
        x_2 = torch.cat((x_2, item.unsqueeze(0)), 0)
    # print('fusion_aug_two_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x_1.size()))

    return x_1, x_2, y

def fusion_aug_one_image(x, y, session, args, alpha=20.0, mix_times=4):  # mixup based
    batch_size = x.size()[0]
    mix_data = []
    mix_target = []

    # So once they give a large alpha the beta

    # print('fusion_aug_one_image | before fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size): # In the scenario where all images in x are from different labels. Each mixup will yield a new generated label. Which means the number of classes will increase to (C * C-1) with 4 samples each
            if y[i] != y[index][i]:
                new_label = fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data:
        x = torch.cat((x, item.unsqueeze(0)), 0)
    # print('fusion_aug_one_image | after fusion | length of the data: {0}, size of image: {1}'.format(len(y), x.size()))

    return x, y

def fusion_aug_generate_label(y_a, y_b, session, args):
    # For example: y_a = 2 y_b = 18
    # current_total_cls_num = 60
    current_total_cls_num = args.base_class + session * args.way
    if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
        y_a, y_b = y_a, y_b
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
        y_a = y_a - (current_total_cls_num - args.way)
        y_b = y_b - (current_total_cls_num - args.way)
        assert y_a != y_b
        if y_a > y_b:  # make label y_a smaller than y_b
            tmp = y_a
            y_a = y_b
            y_b = tmp
        label_index = int(((2 * args.way - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
    return label_index + current_total_cls_num
