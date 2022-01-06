"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from dropnprune import DropNPrune

__all__ = [
    "ResNet",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        mid_planes,
        out_planes,
        resnet_planes,
        stride=1,
        option="A",
        dropnprune_p=0.005,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.dropnprune1 = DropNPrune(dropnprune_p, mid_planes)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.dropnprune2 = DropNPrune(dropnprune_p, out_planes, resnet_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != resnet_planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """

                def shortcut_fn(x):
                    sel_h = torch.arange(0, x.shape[2], 2)
                    sel_w = torch.arange(0, x.shape[3], 2)
                    xx = x[:, :, :, sel_w]
                    xx = xx[:, :, sel_h]
                    return F.pad(
                        xx,
                        (0, 0, 0, 0, resnet_planes // 4, resnet_planes // 4),
                        "constant",
                        0,
                    )

                self.shortcut = LambdaLayer(shortcut_fn)
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * resnet_planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * resnet_planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropnprune1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropnprune2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, planes_per_layer=[None, None, None]
    ):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, planes_per_layer[0], 16, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, planes_per_layer[1], 32, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, planes_per_layer[2], 64, num_blocks[2], stride=2
        )
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, resnet_planes, num_blocks, stride):
        # e.g.: planes = [[7, 8], [6, 5], [3, 9], [16, 16], [12, 14]]
        if planes is None:
            planes = [[resnet_planes, resnet_planes] for _ in range(num_blocks)]
        assert len(planes) == num_blocks
        assert all([len(i) == 2 for i in planes])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride, (mid_planes, out_planes) in zip(strides, planes):
            layers.append(
                block(self.in_planes, mid_planes, out_planes, resnet_planes, stride)
            )
            self.in_planes = resnet_planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean(3).mean(2)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32(planes_per_layer=None):
    if planes_per_layer is None:
        return ResNet(BasicBlock, [5, 5, 5])
    return ResNet(BasicBlock, [5, 5, 5], planes_per_layer=planes_per_layer)


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
