import torch
from torch import nn
import torch.nn.functional as F

import errno
import hashlib
import os
import warnings
import re
import shutil
import sys
import tempfile
from tqdm import tqdm
from urllib.request import urlopen
from urllib.parse import urlparse  # noqa: F401

def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim

        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x


class Model(nn.Module):
    def __init__(self, args, pretrained=False, progress=True):
        super(Model, self).__init__()
        self.backbone = Model.get_backbone(args.dataset, args.arch)
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        if pretrained:
            print('Model | load pre-trained model.')
            model_dict = self.backbone.state_dict()
            state_dict = load_state_dict_from_url(model_urls[args.arch], progress=progress)
            state_dict = {k: v for k, v in state_dict.items() if k not in ['fc.weight', 'fc.bias']}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        if args.no_projector:
            self.encoder = nn.Sequential(
                self.backbone
            )
        else:
            self.projector = projection_MLP(out_dim, args.feat_dim, args.num_proj_layers)
            self.encoder = nn.Sequential(
                self.backbone,
                self.projector
            )

        total_num_of_cls = args.num_classes
        if args.data_fusion:
            aug_for_base = int(((args.base_class) * (args.base_class - 1))/2)
            aug_for_inc = int(((args.way) * (args.way - 1))/2)
            # print('Model | aug_for_base: {0}, aug_for_inc: {1}'.format(aug_for_base, aug_for_inc))
            aug_num_of_cls =  aug_for_base + (args.sessions -1) * aug_for_inc
            total_num_of_cls = total_num_of_cls + aug_num_of_cls
        # print('Model | total_num_of_cls: {0}'.format(total_num_of_cls))

        if args.no_projector:
            self.angular_fc = nn.Linear(out_dim, total_num_of_cls, bias=False)
        else:
            self.angular_fc = nn.Linear(args.feat_dim, total_num_of_cls, bias=False)
        nn.init.xavier_uniform_(self.angular_fc.weight)

    @staticmethod
    def get_backbone(dataset, backbone_name):
        if dataset == 'cifar100' or dataset == 'mini_imagenet':
            from .resnet_CIFAR import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        elif dataset == 'cub200':
            from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152  # standard ResNet
        else:
            raise RuntimeError('Something is wrong.')

        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152()}[backbone_name]

    def forward(self, image):
        encoder_feature = self.encoder(image)
        return encoder_feature


    def get_angular_output(self, images):
        encoder_feature = self.encoder(images)
        wf = F.linear(F.normalize(encoder_feature, p=2, dim=1), F.normalize(self.angular_fc.weight, p=2, dim=1))
        return wf










