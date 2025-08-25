import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUB200(Dataset):
    def __init__(self, root='./', train=True,
                 index_path=None, index=None, base_sess=None, 
                 rand_aug = False, color_jitter = False, rot_transform=False,
                 transform = None, dino_transform = None): # Extra parameters
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self._pre_operate(self.root)
        self.rot_transform = rot_transform
        self.dino_transform = dino_transform

        if train:
            if not rand_aug:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                if color_jitter:
                    self.transform.transforms.insert(2, transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            else:
                self.transform = transforms.Compose([
                    transforms.RandAugment(num_ops = 2),
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)

            if transform is not None:
                self.transform = transform  # Overwriting with user provided transform 

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                # if index is not None:
                #     # Note: Index contains list of sessions required
                #     self.data, self.targets = self.SelectfromTxts(self.data2label, index)
                # else:
                #     self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                # self.data, self.targets = self.SelectfromTxtOffset(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxts(self, data2label, index):
        """
            Selects data and targets from several index paths (indicated by list supplied in the index field)
        """
        index_paths = ["data/index_list/cub200/session_" + str(i) + '.txt' for i in index]
        data_tmp = []
        targets_tmp = []
        for path in index_paths:
            path_indices = open(path).read().splitlines()
            for i in path_indices:
                img_path = os.path.join(self.root, i)
                data_tmp.append(img_path)
                targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromTxtOffset(self, data2label, index_path, offset=100):
        session = int(index_path.split(".")[0].split("/")[-1].split("_")[-1])
        starting_index = 100 + ((session-2)*10)
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        class2id = {}
        unique_classes = 0
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            class_label = data2label[img_path]
            if class_label not in class2id: 
                class2id[class_label] = starting_index + unique_classes
                unique_classes += 1
            targets_tmp.append(class2id[class_label])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)    
        
    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        if self.rot_transform:
            img0 = Image.open(path).convert('RGB')
            
            # TODO: Maybe its better to randomly rotate and send the rotation label instead of inside a batch. That might be making the problem worse:
            rot_label = torch.randint(4, size=(1,)).item()
            rot_img = self.transform(transforms.functional.rotate(img0, 90 * rot_label, expand=True))
            orig_img = self.transform(img0)
            return orig_img, rot_img, rot_label, targets

            # return torch.stack(images, dim=0), rotation_labels, targets

        img0 = Image.open(path).convert('RGB')
        image = self.transform(img0)

        if self.dino_transform is not None:
            crops = self.dino_transform(img0)
            return image, targets, crops
        
        return image, targets



if __name__ == '__main__':
    txt_path = "../../data/index_list/cub200/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '~/dataloader/data'
    batch_size_base = 400
    trainset = CUB200(root=dataroot, train=False,  index=class_index,
                      base_sess=True)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)

    # txt_path = "../../data/index_list/cifar100/session_2.txt"
    # # class_index = open(txt_path).read().splitlines()
    # class_index = np.arange(base_class)
    # trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
    #                     base_sess=True)
    # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
