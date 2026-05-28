"""Reference architectures for SPRKD's Experiment 2 (TinyImageNet).

The CIFAR-style ResNet implementation below is the one used in
``EXPERIMENTAL_MODEL_EVALUATIONS.ipynb`` (the canonical Colab for the
TinyImageNet runs reported in the SPRKD paper). It matches Heimann's
``pytorch_resnet_cifar10`` repository:

    https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

Differences from the standard torchvision ``ResNet``:

* Initial convolution is ``Conv2d(3 -> 16, 3x3, stride=2, padding=1)``
  rather than ``7x7, stride=2`` followed by maxpool. This is the standard
  CIFAR/TinyImageNet recipe.
* Residual shortcuts use option ``"A"`` (zero-padded identity, no extra
  parameters) by default - identical to the canonical Colab and Heimann's
  CIFAR ResNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = [
    "BasicBlock",
    "LambdaLayer",
    "ResNetCIFAR",
    "build_resnet20",
    "build_resnet32",
    "build_resnet44",
    "build_resnet56",
]


def _weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """Trivial wrapper used by option-A residual shortcuts."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    """CIFAR-style basic residual block (no bottleneck)."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        option: str = "A",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                pad = planes // 4
                self.shortcut = LambdaLayer(
                    lambda x, pad=pad: F.pad(
                        x[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad), "constant", 0
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                raise ValueError(f"unknown shortcut option: {option!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    """CIFAR-style ResNet (Heimann 2018; canonical SPRKD Colab)."""

    def __init__(
        self,
        block: type = BasicBlock,
        num_blocks=(3, 3, 3),
        num_classes: int = 200,
        in_channels: int = 3,
        first_stride: int = 2,
    ):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=first_stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)


# Convenience constructors mirroring the canonical Colab depths.

def build_resnet20(num_classes: int = 200, **kwargs) -> ResNetCIFAR:
    """ResNet-20 used in ``EXPERIMENTAL_MODEL_EVALUATIONS.ipynb`` for TinyImageNet."""

    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)


def build_resnet32(num_classes: int = 200, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes, **kwargs)


def build_resnet44(num_classes: int = 200, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [7, 7, 7], num_classes=num_classes, **kwargs)


def build_resnet56(num_classes: int = 200, **kwargs) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes, **kwargs)
