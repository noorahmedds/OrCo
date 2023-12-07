import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
import torch.nn.functional as F
import copy
import numpy as np
import torchvision.models as models
from collections import defaultdict

# from utils.arg_parse import opt
# from utils.logging_setup import logger

__all__ = ['ResNet', 'resnet18']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # if opt.res18_larger:
        #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # else:
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        # if not opt.res18_larger:
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)

        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def stem_param(self):
        for name, param in self.named_parameters():
            if 'layer4' in name:
                yield param
            # if 'layer3' in name:
            #     yield param

    def stem_param_named(self):
        for name, param in self.named_parameters():
            if 'layer4' in name:
                yield name, param

    def get_pretrained_weights(self):
        model = models.resnet18(pretrained=True)
        img_net_dict = model.state_dict()
        load_stat = self.load_state_dict(img_net_dict, strict=False)
        logger.debug(load_stat)
        del model


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

class MLP(nn.Module):
    def __init__(self, checkpoint_init=None):
        super().__init__()

        self.encoder = resnet18()
        self.encoder.get_pretrained_weights()

        # self.encoder = models.resnet18(pretrained=True)
        self.i_dim = 512

        self.fc = nn.Linear(self.i_dim, opt.n_base_classes, bias=opt.bias_classifier)
        self.scale = None

        if opt.cosine_sim:
            assert not opt.bias_classifier
            # self.scale = nn.Parameter(torch.FloatTensor([10]))

        self.gfsv = True
        self.old_model = None
        self.distill_model = None
        self.base_centering = False
        self.base_center_representation = None
        self.fc_novel = None
        self._state_dict = None
        self.checkpoint_init = checkpoint_init
        self.fc_sessions = None
        self.imprint_init = False
        self.imprint_weights_dict = defaultdict(lambda: torch.zeros(self.i_dim, 1).to(opt.device))
        if opt.imprint_w:
            self.imprint_init = True

    def fs_init(self):
        self.gfsv = False
        self.fc_novel = nn.Linear(self.i_dim, opt.n_way, bias=opt.bias_classifier)

    def new_ses_fc(self, dim):
        if self.fc_sessions is None:
            self.fc_sessions = nn.ModuleList([nn.Linear(self.i_dim, dim, bias=opt.bias_classifier)])
        else:
            self.fc_sessions.append(nn.Linear(self.i_dim, dim, bias=opt.bias_classifier))
        self.fc_sessions.to(opt.device)

    def update_checkpoint_init(self):
        self.checkpoint_init = copy.deepcopy(self.state_dict())

    def set_imprint_init_true(self):
        self.imprint_init = True

    def set_imprint_init_false(self):
        self.imprint_init = False

    def L2_weight_loss(self):
        loss = None
        for name, param in self.encoder.stem_param_named():
            if loss is None:
                loss = (self.checkpoint_init['encoder.' + name] - param).pow(2).sum()
            else:
                loss += (self.checkpoint_init['encoder.' + name] - param).pow(2).sum()

        # if 'fc_novel.weight' in self.checkpoint_init:

        return loss * opt.l2_weight
        # return loss * 1e+4

    def forward(self, input):
        x = input['data']
        output = {}
        x = x.to(device=opt.device, non_blocking=True)
        x = self.encoder(x)
        if self.base_centering:
            return {'x_base_embed': x}

        if self.base_center_representation is not None:
            x -= self.base_center_representation

        if opt.cosine_sim:
            x = F.normalize(x)
            if opt.cross_c or opt.triplet_loss or opt.knn_embed:
                output.update({'x_embed': x})
            x = self.scale * x
            if self.training:
                if self.gfsv:
                    self.fc.weight.data = F.normalize(self.fc.weight.data)
                else:
                    self.fc_novel.weight.data = F.normalize(self.fc_novel.weight.data)
        else:
            if opt.cross_c or opt.triplet_loss or opt.knn_embed:
                output.update({'x_embed': x})


        if opt.imprint_w:
            x = F.normalize(x)
            if self.imprint_init:
                for enum_idx, vid_label in enumerate(input['label']):
                    # assert int(vid_lab) < opt.n_way
                    self.imprint_weights_dict[int(vid_label)] += x[enum_idx].reshape(-1, 1)
                return output

        if opt.fscil:
            if opt.gce:
                probs = self.fc(x)
                for ly in self.fc_sessions:
                    probs2 = ly(x)
                    probs = torch.cat((probs, probs2), dim=1)
            else:
                probs = self.fc_sessions[-1](x)
        else:
            if self.gfsv:
                probs = self.fc(x)
            else:
                # x = self.scale_novel +  x
                probs = self.fc_novel(x)


        output.update({'probs': probs})
        return output

    def finilaze_iw(self, label_bias):
        assert opt.fscil
        for key, val in self.imprint_weights_dict.items():
            if key < label_bias[1]:
                # pass
                self.fc.weight.data[key] = val.squeeze() / opt.k_shot
            else:
                ses_idx = 0
                bias = label_bias[1]
                key_spare = key
                while key >= bias:
                    ses_idx += 1
                    # key -= bias
                    bias = label_bias[ses_idx + 1]
                # key = key % opt.n_way
                # key = key_spare - bias
                key -= bias
                self.fc_sessions[ses_idx-1].weight.data[key] = val.squeeze() / opt.k_shot
        self.fc.weight.data = F.normalize(self.fc.weight.data)
        for ses_idx in range(opt.n_ses-1):
            self.fc_sessions[ses_idx].weight.data = F.normalize(self.fc_sessions[ses_idx].weight.data)
        self.imprint_init = False

    def stem_param(self):
        for param in self.encoder.stem_param():
            yield param

    def fc_param(self):
        for param in self.fc.parameters():
            yield param
        if opt.fscil:
            for ly in self.fc_sessions:
                for param in ly.parameters():
                    yield param
        else:
            for param in self.fc_novel.parameters():
                yield param

    def base_cl_param(self):
        for param in self.fc.parameters():
            yield param
        if opt.fscil:
            for i in range(len(self.fc_sessions)-1):
                for param in self.fc_sessions[i].parameters():
                    yield param


    def novel_cl_param(self):
        if opt.fscil:
            for param in self.fc_sessions[-1].parameters():
                yield param
        else:
            for param in self.fc_novel.parameters():
                yield param

    # def novel_fscil_cl_param(self):
    #     if opt.fscil:
    #         for param in self.fc_sessions[-1].parameters():
    #             yield param
    #     else:
    #         for param in self.fc_novel.parameters():
    #             yield param


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, input, lymbda=1):
        probs = output['probs'].to(opt.device)
        target = input['label'].to(opt.device)
        if probs.size(0) > 1:
            target = target.squeeze()

        return lymbda * F.cross_entropy(probs, target)


class DistillKL(nn.Module):
    """KL divergence for distillation
    from https://github.com/WangYueFt/rfs/blob/master/distill/criterion.py"""

    def __init__(self):
        super(DistillKL, self).__init__()
        self.T = opt.distill_T

    def forward(self, y_s, y_t):
        y_s = y_s['probs'].to(opt.device)
        y_t = y_t['probs'].to(opt.device)
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


def create_model(**kwargs):

    model = MLP()

    if torch.cuda.is_available():
        opt.device = 'cuda'
    else:
        opt.device = 'cpu'

    if opt.one_plus2_model.endswith('pth.tar'):
        if opt.device == 'cpu':
            checkpoint = torch.load(opt.one_plus2_model, map_location='cpu')['state_dict']
        else:
            # checkpoint = torch.load(opt.one_plus2_model)['state_dict']
            checkpoint = torch.load(opt.one_plus2_model)['params']
        a = model.load_state_dict(checkpoint, strict=True)
        logger.debug(a)


    loss = CrossEntropyLoss()

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay)

    if opt.optim == 'sgd':
        lr = opt.lr if not opt.lr_warmup else 1e-8
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=True
                                    )
    model.to(opt.device)
    logger.debug(str(model))
    for name, param in model.named_parameters():
        logger.debug('%s\n%s' % (str(name), str(param.norm())))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer