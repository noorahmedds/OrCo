import random
from tqdm import tqdm

import numpy as np
from PIL import ImageFilter

import torch
import torchvision

from sklearn.linear_model import LogisticRegression

from . import pytorch_utils as ptu



# standard imagenet stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def accuracy_score(preds, labels):
    """ a simple top1 accuracy scoring function """
    if isinstance(preds, np.ndarray):
        return float((preds == labels).mean())
    else:
        return float((preds == labels).float().mean())


class MyCheckpoint(ptu.Checkpoint):
    """ an adaptation of ptu.Checkpoint for MoCo overriding batch_pass and agg_results"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def batch_pass(self,
                   device,
                   batch,
                   train,
                   *args, **kwargs):

        results = {}
        pbar_postfix = {}
        
        if self.model.moco:
            (q_img, k_img), labels = batch
            q_img = q_img.to(device)
            k_img = k_img.to(device)
            labels = labels.to(device)
            self.batch_size = q_img.shape[0]

            q, logits, zeros = self.model(q_img, k_img)

            loss = self.criterion(logits.float(), zeros.long())

            results['q'] = q.detach().cpu().numpy()
            results['labels'] = labels.detach().cpu().numpy()
        else:
            img, labels = batch
            img = img.to(device)
            labels = labels.to(device)
            self.batch_size = img.shape[0]

            out = self.model(img)

            loss = self.criterion(out.float(), labels.long())

            results['out'] = out.argmax(dim=1).detach().cpu().numpy()
            results['labels'] = labels.detach().cpu().numpy()
            pbar_postfix['score'] = self.score(results['labels'], results['out'])
            if len(self.raw_results) > 0:
                pbar_postfix['avg_score'] = self.score(np.concatenate(self.raw_results['labels']), np.concatenate(self.raw_results['out']))

        return loss, results, pbar_postfix

    def agg_results(self, results, train):
        single_num_score = None
        additional_metrics = {}

        if self.model.moco:
            q = np.concatenate(results['q'])
            labels = np.concatenate(results['labels'])

            if train:
                self.model.clf = LogisticRegression(**self.model.clf_hyperparams)
                self.model.clf.fit(q, labels)

            preds = self.model.clf.predict(q)
            single_num_score = self.score(labels, preds)
        else:
            preds = np.concatenate(results['out'])
            labels = np.concatenate(results['labels'])

            single_num_score = self.score(labels, preds)

        return single_num_score, additional_metrics


class Dataset(torch.utils.data.Dataset):
    """ a Dataset class for preloading data into memory """
    def __init__(self,
                 path: str,
                 transforms: torchvision.transforms.Compose,
                 preload_data: bool=False,
                 tqdm_bar: bool=False):
        """
        path : str
        """
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.preload_data = preload_data
        self.torchvision_dataset = torchvision.datasets.ImageFolder(path)

        if self.preload_data:
            self.images = []
            self.labels = []

            if tqdm_bar:
                pbar = tqdm(self.torchvision_dataset)
            else:
                pbar = self.torchvision_dataset

            for image, label in pbar:
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.torchvision_dataset)

    def __getitem__(self, i):
        if self.preload_data:
            img = self.transforms(self.images[i])
            l = self.labels[i]
        else:
            img, l = self.torchvision_dataset.__getitem__(i)
            img = self.transforms(img)
        return img, l


class Config:
    """ a simple class for managing experiment setup """
    def __call__(self):
        return vars(self)

    def __repr__(self):
        return str(self())

    def __str__(self):
        return self.__repr__()


class TwoCropsTransform:
    """ twice applied transforms to an image """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transforms(x), self.transforms(x)

    def __repr__(self):
        return str(self.transforms)

    def __str__(self):
        return self.__repr__()


class GaussianBlur:
    """ apply ImageFilter.GaussianBlur to an image """
    def __init__(self, sigma1=0.1, sigma2=2.0):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self, x):
        return x.filter(ImageFilter.GaussianBlur(random.uniform(self.sigma1, self.sigma2)))

    def __repr__(self):
        return f'GaussianBlur({self.sigma1}, {self.sigma2})'

    def __str__(self):
        return self.__repr__()


moco_v1_transforms = TwoCropsTransform(torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
]))


moco_v2_transforms = TwoCropsTransform(torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), ], p=0.8),
    torchvision.transforms.RandomGrayscale(p=0.2),
    torchvision.transforms.RandomApply([GaussianBlur(sigma1=0.1, sigma2=2.0), ], p=0.5),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
]))


clf_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


clf_val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


