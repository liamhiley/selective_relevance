import sys
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
# spec = importlib.util.spec_from_file_location("torchexplain", "/home/hileyl/Projects/DAIS-ITA/Remote/torchexplain/torchexplain/__init__.py")
# torchexplain = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(torchexplain)
from .. import torchexplain

import pdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
save_path ="/home/hileyl/Projects/sel_rel/sR_results/downsample_resizing/kinetics400/trilinear/test/"

"""
author: https://github.com/kenshohara/3D-ResNets-PyTorch/
"""

def get_inplanes():
    return [64, 128, 256, 512]

def get_X_inplanes():
    return [128, 256, 512, 1024]

def conv3x3x3(in_planes, out_planes, stride=1, lib=torchexplain):
    return lib.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1, lib=torchexplain):
    return lib.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, train=False, **kwargs):
        super().__init__()
        if train:
            self.lib = nn
        else:
            self.eval()
            self.lib = torchexplain

        self.conv1 = conv3x3x3(in_planes, planes, stride, lib=self.lib)
        self.bn1 = self.lib.BatchNorm3d(planes)
        self.relu = self.lib.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, lib=self.lib)
        self.bn2 = self.lib.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.training:
            out += residual
        else:
            out = self.lib.add(out,residual)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, train=False, **kwargs):
        super().__init__()
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain

        self.conv1 = conv1x1x1(in_planes, planes, lib=self.lib)
        self.bn1 = self.lib.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, lib=self.lib)
        self.bn2 = self.lib.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, lib=self.lib)
        self.bn3 = self.lib.BatchNorm3d(planes * self.expansion)
        self.relu = self.lib.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.training:
            out += residual
        else:
            out = self.lib.add(out,residual)
        out = self.relu(out)

        return out

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, cardinality, stride=1,
                 downsample=None, train=False):
        super(ResNeXtBottleneck, self).__init__()
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = self.lib.Conv3d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = self.lib.BatchNorm3d(mid_planes)
        self.conv2 = self.lib.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = self.lib.BatchNorm3d(mid_planes)
        self.conv3 = self.lib.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = self.lib.BatchNorm3d(planes * self.expansion)
        self.relu = self.lib.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # if x.shape == torch.Size([2,512,16,32,32]):
        #     pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.training:
            out += residual
        else:
            out += residual
            # out = self.lib.add(out,residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 in_planes=None,
                 channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 cardinality=None,
                 widen_factor=1.0,
                 num_classes=400,
                 range=None,
                 train=False,
                 **kwargs):
        super().__init__()
        if train:
            self.lib = nn
        else:
            self.lib = torchexplain

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        if not in_planes:
            self.in_planes = block_inplanes[0]
        else:
            self.in_planes = in_planes
        self.no_max_pool = no_max_pool

        if train:
            self.conv1 = self.lib.Conv3d(channels,
                                   self.in_planes,
                                   kernel_size=(conv1_t_size, 7, 7),
                                   stride=(conv1_t_stride, 2, 2),
                                   padding=(conv1_t_size // 2, 3, 3),
                                   bias=False)
        else:
            self.conv1 = self.lib.Conv3d(channels,
                                         self.in_planes,
                                         kernel_size=(conv1_t_size, 7, 7),
                                         stride=(conv1_t_stride, 2, 2),
                                         padding=(conv1_t_size // 2, 3, 3),
                                         bias=False,
                                         range=range
            )
        self.bn1 = self.lib.BatchNorm3d(self.in_planes)
        self.relu = self.lib.ReLU(inplace=True)
        self.maxpool = self.lib.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, cardinality=cardinality,
                                       train=train)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       cardinality=cardinality,
                                       stride=2,
                                       train=train)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       cardinality=cardinality,
                                       stride=2,
                                       train=train)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       cardinality=cardinality,
                                       stride=2,
                                       train=train)

        self.avgpool = self.lib.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = self.lib.Linear(block_inplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, self.lib.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, self.lib.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        if self.lib == torchexplain:
            F = self.lib
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality=None, stride=1, train=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride, lib=self.lib),
                    self.lib.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  cardinality=cardinality,
                  stride=stride,
                  downsample=downsample,
                  train=train))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, cardinality=cardinality, train=train))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # HERE
        x = self.layer4(x)


        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, in_planes=None, cardinality=None, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if cardinality:
        if model_depth == 50:
            model = ResNet(ResNeXtBottleneck, [3, 4, 6, 3], get_X_inplanes(), in_planes=in_planes, cardinality=cardinality, **kwargs)
        elif model_depth == 101:
            model = ResNet(ResNeXtBottleneck, [3, 4, 23, 3], get_X_inplanes(), in_planes=in_planes, cardinality=cardinality, **kwargs)
        elif model_depth == 152:
            model = ResNet(ResNeXtBottleneck, [3, 8, 36, 3], get_X_inplanes(), in_planes=in_planes, cardinality=cardinality, **kwargs)
    else:
        if model_depth == 10:
            model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
        elif model_depth == 18:
            model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
        elif model_depth == 34:
            model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
        elif model_depth == 50:
            model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
        elif model_depth == 101:
            model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
        elif model_depth == 152:
            model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
        elif model_depth == 200:
            model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model
