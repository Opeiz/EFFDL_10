import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
import numpy as np


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1
    @classmethod
    def set_factorization_level(cls,fact_level = 0):
        cls.fact_level = fact_level
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.fact_level == 0:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.fact_level == 1:
            self.conv_dw = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
            self.bn_dw = nn.BatchNorm2d(in_planes)
            self.conv_pw = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_pw = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.fact_level == 2:
            self.conv_dw = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
            self.bn_dw = nn.BatchNorm2d(in_planes)
            self.conv_pw = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_pw = nn.BatchNorm2d(planes)
            self.conv2_dw = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
            self.bn2_dw = nn.BatchNorm2d(planes)
            self.conv2_pw = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        if self.fact_level == 0:
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
        if self.fact_level == 1:
            out = self.conv_dw(out)
            out = self.bn_dw(out)
            out = self.conv_pw(out)
            out = self.conv2(F.relu(self.bn_pw(out)))
        if self.fact_level == 2:
            out = self.conv_dw(out)
            out = self.bn_dw(out)
            #out = self.conv_pw(out)
            #out = self.conv2_dw(F.relu(self.bn_pw(out)))
            out = F.relu(self.bn_pw(self.conv_pw(out)))
            out = self.conv2_dw(out)
            out = self.bn2_dw(out)
            out = self.conv2_pw(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18_fact(fact_level):
    PreActBlock.set_factorization_level(fact_level=fact_level)
    return PreActResNet(PreActBlock, [1,1,1,1])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


# def test(fact_level):
#     net = PreActResNet18(fact_level)
#     y = net((torch.randn(1,3,32,32)))
#     print(y.size())


