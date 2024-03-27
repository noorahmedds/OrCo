import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
import utils
from dataloader.autoaugment import CIFAR10Policy, Cutout, ImageNetPolicy

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'mini_imagenet1s':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 1
        args.sessions = 9
    args.Dataset=Dataset
    if args.exemplars_count == -1:
        args.exemplars_count = args.shot
    assert args.exemplars_count <= args.shot, "Exemplars count cannot be greater than the number of shots in your few shot data"
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_rot_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True, rand_aug = args.rand_aug, rot_transform = True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index, rot_transform = True)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, data_aug = args.data_aug)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)


    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader(args, debug = False, dino_transform = None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class if not debug else 5) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True, 
                                       rand_aug = args.rand_aug, 
                                       dino_transform = dino_transform)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, data_aug = args.data_aug)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)


    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def appendKBaseExemplars(dataset, args, nclass, path2conf = None):
    """
        Take only labels from args.base_class and in data self.data append the single exemplar
    """
    if args.dataset == "cifar100":
        # Get dataset indices under base_class
        for i in range(nclass):
            ind_cl = np.where(i == dataset.targets_all)[0]

            # Choose top 5 from ind_cl and append into data_tmp (done to stay consistent across experiments)
            ind_cl = ind_cl[:args.exemplars_count]
            # ind_cl = ind_cl[:1]

            dataset.data = np.vstack((dataset.data, dataset.data_all[ind_cl]))
            dataset.targets = np.hstack((dataset.targets, dataset.targets_all[ind_cl]))

        return


    label2data = {}
    for k,v in dataset.data2label.items():
        if v < nclass:
            if v not in label2data: label2data[v] = []
            label2data[v].append(k)
    
    n_samples = len(label2data[0])

    # sample_ix = torch.randint(n_samples, (args.base_class,)) 
    # To maintain simplicity and the reduce added complexity we always sample the first K exemplars from the base class.
    # This should ideally not introduce any biases

    data_tmp = []
    targets_tmp = []

    if path2conf is not None:
        for i in range(nclass):    
            # Add only the paths which in the sorted order of confidence
            # i.e. get label2data[i][k] where k belongs to the path which produced the least confidence for this class
            confs_sorted_ix = np.argsort(np.array(path2conf[i]["conf"]))
            for k in range(args.exemplars_count):
                path_ix = confs_sorted_ix[k]
                data_tmp.append(path2conf[i]["path"][path_ix])
                targets_tmp.append(i)
    else:
        for i in range(nclass):
            for k in range(args.exemplars_count):
                data_tmp.append(label2data[i][k])
                targets_tmp.append(i)
            # data_tmp.append(label2data[i][0])
            # targets_tmp.append(i)

    dataset.data.extend(data_tmp)
    dataset.targets.extend(targets_tmp)

    return data_tmp, targets_tmp

def get_jointloader(args, session, dino_transform = None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, keep_all = True)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path, dino_transform = dino_transform, rand_aug = args.rand_aug_joint)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    
    if args.train_inter:
        nclass = args.base_class + (args.way * (session-1))
    else:
        nclass = args.base_class
        
    # From the session 0 index list take a label and image path and append to the train set. The rest should be handled automatically
    data_previous, targets_previous = appendKBaseExemplars(trainset, args, nclass)
    
    if args.batch_size_joint == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True,            # <<< TODO: Shuffled. Check if this is problematic
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
        #                                           num_workers=args.num_workers, pin_memory=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_joint, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # Make previous data loader
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        prevset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                         index=class_index, base_sess=False, keep_all = True)
    if args.dataset == 'cub200':
        # prevset = args.Dataset.CUB200(root=args.dataroot, train=True,
        #                                index_path=txt_path, dino_transform = dino_transform, rand_aug = args.rand_aug_joint)
        prevset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=np.arange(5))
    if args.dataset == 'mini_imagenet':
        # TODO: Fix
        prevset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                       index_path=txt_path)
    
    prevset.data = data_previous
    prevset.targets = targets_previous
    prevloader = torch.utils.data.DataLoader(dataset=prevset, batch_size=1, shuffle=True,num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, prevloader

def get_strong_transform(args):
    if args.dataset == 'cifar100':
        strong_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
    
    elif args.dataset == 'mini_imagenet':
        strong_transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.RandomResizedCrop(84, scale=(args.min_crop_scale, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
    elif args.dataset == "cub200":
        strong_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])


    return strong_transform

def get_jointloader_fixed(args, session, dino_transform = None):
    transform = None
    if args.strong_transform:
        transform = get_strong_transform(args)

    # Update joint loader such that all previous images from the intermediate sessions and the 
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        dataset_class = args.Dataset.CIFAR100
        trainset = dataset_class(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, keep_all = True, transform = transform)
    if args.dataset == 'cub200':
        dataset_class = args.Dataset.CUB200
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, dino_transform = dino_transform, rand_aug = args.rand_aug_joint, transform = transform)
    if args.dataset == 'mini_imagenet':
        dataset_class = args.Dataset.MiniImageNet
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = transform)
    
    # if args.train_inter:
    #     nclass = args.base_class + (args.way * (session-1))
    # else:
    nclass = args.base_class

    if args.train_inter:
        # Now for each previous session i.e. session > 0 and session < curr_session
        # load trainset using the index files. And append data and labels from this dataset to the current one
        for inter_ix in range(1, session): # is = intermediate_sessino
            txt_path = "data/index_list/" + args.dataset + "/session_" + str(inter_ix + 1) + '.txt'
            if args.dataset == "cifar100":
                class_index = open(txt_path).read().splitlines()
                inter_set = dataset_class(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False)   # Get data from current index    

                trainset.data = np.vstack((trainset.data, inter_set.data))
                trainset.targets = np.hstack((trainset.targets, inter_set.targets))
            else:
                inter_set = dataset_class(root=args.dataroot, train=True, index_path=txt_path, base_sess=False)   # Get data from current index
                # Append the new data from the previous intermeidate sessions to the current dataset

                trainset.data.extend(inter_set.data)
                trainset.targets.extend(inter_set.targets)
        
    # Append the base classes to the current dataset
    appendKBaseExemplars(trainset, args, nclass)

    if args.batch_size_joint == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True,            # <<< TODO: Shuffled. Check if this is problematic
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
        #                                           num_workers=args.num_workers, pin_memory=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_joint, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader

def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class MultiCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, n_views = 2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        out = []
        for i in range(self.n_views):
            out.append(self.transform(x))
        return out

from torchvision import transforms

def get_supcon_dataloader(args):
    
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        if args.rand_aug_sup_con:
            # train_transform = transforms.Compose([
            #     transforms.RandAugment(num_ops = 6),
            #     transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
            #     transforms.ToTensor(),
            #     normalize,
            # ])    
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=True, transform = MultiCropTransform(train_transform))
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=False)

    if args.dataset == 'cub200':
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if args.rand_aug_sup_con:
            # train_transform = transforms.Compose([
            #     transforms.RandAugment(num_ops = 6),
            #     transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
            #     transforms.ToTensor(),
            #     normalize,
            # ])   
            train_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #add autoaug
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True, 
                                       rand_aug = args.rand_aug, 
                                       dino_transform = None, transform = MultiCropTransform(train_transform))
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandAugment(num_ops = 3, magnitude=11),
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, data_aug = args.data_aug, 
                                             transform = TwoCropTransform(train_transform))
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_sup_con, shuffle=True,
                                              num_workers=8, pin_memory=True, drop_last = args.drop_last_batch)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_previous_novel_dataloader(args, session):
    if session == 1:
        return None, None, None
    # session refers to uptil and including this session
    if args.dataset == 'cifar100':
        raise NotImplementedError("Function not impl for cifar100")
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=list(range(1,session-1)))
    if args.dataset == 'mini_imagenet':
        raise NotImplementedError("Function not impl for cifar100")
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes 
    class_new = np.arange(args.base_class + (session-1) * args.way)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args, session, dino_transform = None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path, dino_transform = dino_transform, rand_aug = args.rand_aug_novel)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


def get_supcon_joint_dataloader(args, session, path2conf):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_index = np.arange(args.base_class) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':        
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=args.prob_color_jitter),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        class_index = open(txt_path).read().splitlines()
        dataset_class = args.Dataset.CIFAR100

        # strong_target = None
        # if args.heavy_inter_aug and session > 1:
        #     # Strong target could be a dictiornary where each target is assigned the magnitude of augmentation
        #     # Depending on the target the magnitude of rand augment will be higher
        #     strong_target = {}
        #     for i in range(args.base_class, args.base_class + (args.way * (session-1))):
        #         strong_target[i] = (session - ((i - args.base_class) % args.way)) - 1

        base_aug_mag = 0
        if args.heavy_inter_aug:
             base_aug_mag = session + 2
             
        trainset = dataset_class(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, keep_all = True, transform = MultiCropTransform(train_transform, args.supcon_views),
                                         base_aug_mag=base_aug_mag)

    if args.dataset == 'cub200':
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    #add autoaug
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=args.prob_color_jitter),  
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        dataset_class = args.Dataset.CUB200
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = MultiCropTransform(train_transform, args.supcon_views))

    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.RandAugment(num_ops = 3),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=args.prob_color_jitter),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        dataset_class = args.Dataset.MiniImageNet
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = MultiCropTransform(train_transform, args.supcon_views))

    nclass = args.base_class

    if args.train_inter:
        # Now for each previous session i.e. session > 0 and session < curr_session
        # load trainset using the index files. And append data and labels from this dataset to the current one
        # Add the ability to choose the number of exemplars from previous sessions
        for inter_ix in range(1, session): # is = intermediate_sessino
            txt_path = "data/index_list/" + args.dataset + "/session_" + str(inter_ix + 1) + '.txt'
            if args.dataset == "cifar100":
                class_index = open(txt_path).read().splitlines()
                inter_set = dataset_class(root=args.dataroot, train=True, download=False, index=class_index, base_sess=False)   # Get data from current index    
                # TODO: Add exemplar control here
                trainset.data = np.vstack((trainset.data, inter_set.data))
                trainset.targets = np.hstack((trainset.targets, inter_set.targets))
            else:
                inter_set = dataset_class(root=args.dataroot, train=True, index_path=txt_path, base_sess=False)   # Get data from current index

                if args.exemplars_count != args.shot:
                    # Exemplar Control: Append the new data from the previous intermeidate sessions to the current dataset
                    inter_targets = np.array(inter_set.targets)
                    for i in np.unique(inter_targets):
                        ixs = np.where(inter_targets == i)[0]
                        selected_ixs = list(ixs[:args.exemplars_count])
                        for j in selected_ixs:
                            trainset.data.append(inter_set.data[j])
                            trainset.targets.append(inter_set.targets[j])    
                else:
                    trainset.data.extend(inter_set.data)
                    trainset.targets.extend(inter_set.targets)

    # Append the base classes to the current dataset
    if args.append_hard_positives:
        appendKBaseExemplars(trainset, args, nclass, path2conf)
    else:    
        appendKBaseExemplars(trainset, args, nclass)

    if args.batch_size_joint == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True,            # <<< TODO: Shuffled. Check if this is problematic
                                                  num_workers=args.num_workers, pin_memory=True, drop_last = args.drop_last_batch)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_joint, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True, drop_last = args.drop_last_batch)

    return trainset, trainloader

def get_supcon_dataloader_base_balanced(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandAugment(num_ops = 6),
                transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
                transforms.ToTensor(),
                normalize,
            ])    
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, index=class_index, base_sess=True, keep_all=True, transform = MultiCropTransform(train_transform))

    if args.dataset == 'cub200':
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandAugment(num_ops = 6),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.ToTensor(),
                normalize,
            ])    
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True, 
                                       rand_aug = args.rand_aug, 
                                       dino_transform = None, transform = MultiCropTransform(train_transform))
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandAugment(num_ops = 3, magnitude=11),
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, data_aug = args.data_aug, 
                                             transform = TwoCropTransform(train_transform))
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    targets_len = trainset.targets.shape[0]

    # Remove all data and targets
    appendKBaseExemplars(trainset, args, args.base_class)

    trainset.data = trainset.data[targets_len:]
    trainset.targets = trainset.targets[targets_len:]

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_sup_con, shuffle=True,
                                              num_workers=8, pin_memory=True)
    return trainset, trainloader

def get_supcon_new_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_index = np.arange(args.base_class) # test on a small dataset for debugging purpose
    if args.dataset == 'cifar100':        
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(args.min_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=args.prob_color_jitter),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        class_index = open(txt_path).read().splitlines()
        dataset_class = args.Dataset.CIFAR100

        strong_target = None
        if args.heavy_inter_aug and session > 1:
            # Strong target could be a dictiornary where each target is assigned the magnitude of augmentation
            # Depending on the target the magnitude of rand augment will be higher
            strong_target = {}
            for i in range(args.base_class, args.base_class + (args.way * (session-1))):
                strong_target[i] = (session - ((i - args.base_class) % args.way)) - 1

        trainset = dataset_class(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, keep_all = True, transform = MultiCropTransform(train_transform, args.supcon_views),
                                         strong_target = strong_target)

    if args.dataset == 'cub200':
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(args.min_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=args.prob_color_jitter),  
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        dataset_class = args.Dataset.CUB200
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = MultiCropTransform(train_transform, args.supcon_views))

    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.rand_aug_sup_con:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.RandAugment(num_ops = 3),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=84, scale=(args.min_crop_scale, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=args.prob_color_jitter),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        dataset_class = args.Dataset.MiniImageNet
        trainset = dataset_class(root=args.dataroot, train=True,
                                       index_path=txt_path, transform = MultiCropTransform(train_transform, args.supcon_views))

    if args.batch_size_joint == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=True,            # <<< TODO: Shuffled. Check if this is problematic
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_joint, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader