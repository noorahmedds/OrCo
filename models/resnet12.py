# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

# This part of code is modified from https://github.com/kjunelee/MetaOptNet
# It was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning
# (Oreshkin et al., in NIPS 2018) and A Simple Neural Attentive Meta-Learner
# (Mishra et al., in ICLR 2018).


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class DropBlock(nn.Module):

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: Tensor, gamma: float) -> Tensor:
        # Randomly zeroes 2D spatial blocks of the input tensor.
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1),
                 width - (self.block_size - 1)))
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size(
            )[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask: Tensor) -> Tensor:
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        non_zero_idxes = mask.nonzero()
        nr_blocks = non_zero_idxes.shape[0]

        offsets = torch.stack([
            torch.arange(self.block_size).view(-1, 1).expand(
                self.block_size, self.block_size).reshape(-1),
            torch.arange(self.block_size).repeat(self.block_size),
        ]).t()
        offsets = torch.cat(
            (torch.zeros(self.block_size**2, 2).long(), offsets.long()), 1)
        offsets = offsets.to(mask.device)

        if nr_blocks > 0:
            non_zero_idxes = non_zero_idxes.repeat(self.block_size**2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxes = non_zero_idxes + offsets
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxes[:, 0], block_idxes[:, 1],
                        block_idxes[:, 2], block_idxes[:, 3]] = 1.
        else:
            padded_mask = F.pad(
                mask,
                (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask
        return block_mask

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 drop_rate: float = 0.0,
                 drop_block: bool = False,
                 block_size: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(stride)
        self.downsample = downsample

        # drop block
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x: Tensor) -> Tensor:
        num_batches_tracked = int(self.norm1.num_batches_tracked.cpu().data)
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.max_pool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * num_batches_tracked,
                    1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (
                    feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class ResNet12(nn.Module):
    """ResNet12.

    Args:
        block (nn.Module): Block to build layers. Default: :class:`BasicBlock`.
        with_avgpool (bool): Whether to average pool the features.
            Default: True.
        pool_size (tuple(int,int)): The output shape of average pooling layer.
            Default: (1, 1).
        flatten (bool): Whether to flatten features from (N, C, H, W)
            to (N, C*H*W). Default: True.
        drop_rate (float): Dropout rate. Default: 0.0.
        drop_block_size (int): Size of drop block. Default: 5.
    """

    def __init__(self,
                 block: nn.Module = BasicBlock,
                 with_avgpool: bool = True,
                 pool_size: Tuple[int, int] = (1, 1),
                 flatten: bool = True,
                 drop_rate: float = 0.0,
                 drop_block_size: int = 5) -> None:
        self.inplanes = 3
        super().__init__()

        self.layer1 = self._make_layer(
            block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(
            block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=drop_block_size)
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
            drop_rate=drop_rate,
            drop_block=True,
            block_size=drop_block_size)
        self.with_avgpool = with_avgpool
        if with_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

        self.flatten = flatten
        self.num_batches_tracked = 0
        self.fc = None

    def _make_layer(self,
                    block: nn.Module,
                    out_channels: int,
                    stride: int = 1,
                    drop_rate: float = 0.0,
                    drop_block: bool = False,
                    block_size: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [
            block(self.inplanes, out_channels, stride, downsample,
                  drop_rate, drop_block, block_size)
        ]
        self.inplanes = out_channels * block.expansion

        return nn.Sequential(*layers)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.with_avgpool:
            x = self.avgpool(x)
        if self.flatten:
            x = x.view(x.size(0), -1)
        return x

def resnet12_nc(low_dim=128):
    return ResNet12(BasicBlock)