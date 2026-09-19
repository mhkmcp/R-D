"""CIFAR ResNets, He et al. 2016 §4.2, option-A shortcuts (M1 n=1, M2 n=3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional

from fidnn.models.common import Tap


class ShortcutA(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int):
        super().__init__()
        self.stride = stride
        self.pad = planes - in_planes

    def forward(self, x):
        if self.stride == 1 and self.pad == 0:
            return x
        x = x[:, :, :: self.stride, :: self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tap_pre = Tap()
        self.shortcut = ShortcutA(in_planes, planes, stride)
        self.add = FloatFunctional()
        self.relu_out = nn.ReLU()
        self.tap_out = Tap()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.tap_pre(self.bn2(self.conv2(out)))
        out = self.add.add(out, self.shortcut(x))
        return self.tap_out(self.relu_out(out))


class CifarResNet(nn.Module):
    def __init__(self, n: int, num_classes: int = 10):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.tap_stem = Tap()
        in_planes = 16
        for i, planes in enumerate([16, 32, 64], start=1):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(in_planes, planes, stride=2 if (i > 1 and j == 0) else 1))
                in_planes = planes
            setattr(self, f"layer{i}", nn.Sequential(*blocks))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.tap_pooled = Tap()
        self.fc = nn.Linear(64, num_classes)
        self.tap_logits = Tap()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.tap_stem(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.tap_pooled(self.flatten(self.pool(x)))
        return self.tap_logits(self.fc(x))


def resnet8(num_classes: int = 2) -> CifarResNet:
    return CifarResNet(n=1, num_classes=num_classes)


def resnet20(num_classes: int = 10) -> CifarResNet:
    return CifarResNet(n=3, num_classes=num_classes)
