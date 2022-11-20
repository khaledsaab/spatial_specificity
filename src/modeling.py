import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torchvision.models import DenseNet as _DenseNet
from torchvision.models import ResNet as _ResNet
from torchvision.models.densenet import _load_state_dict
from torchvision.models.densenet import model_urls as densenet_model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls as resnet_model_urls
from torch.hub import load_state_dict_from_url



class ResNet(_ResNet):

    ACTIVATION_DIMS = [64, 128, 256, 512]
    ACTIVATION_WIDTH_HEIGHT = [64, 32, 16, 8]
    RESNET_TO_ARCH = {"resnet18": [2, 2, 2, 2], "resnet50": [3, 4, 6, 3]}

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        if arch not in self.RESNET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.RESNET_TO_ARCH.keys()}"
            )

        block = BasicBlock if arch == "resnet18" else Bottleneck
        super().__init__(block, self.RESNET_TO_ARCH[arch])
        if pretrained:
            state_dict = load_state_dict_from_url(
                resnet_model_urls[arch], progress=True
            )
            self.load_state_dict(state_dict)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512 * block.expansion, num_classes)
        )


class DenseNet(_DenseNet):

    DENSENET_TO_ARCH = {
        "densenet121": {
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
            "num_init_features": 64,
        }
    }

    def __init__(
        self, num_classes: int, arch: str = "densenet121", pretrained: bool = True
    ):
        if arch not in self.DENSENET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.DENSENET_TO_ARCH.keys()}"
            )

        super().__init__(**self.DENSENET_TO_ARCH[arch])
        if pretrained:
            _load_state_dict(self, densenet_model_urls[arch], progress=True)

        self.classifier = nn.Linear(self.classifier.in_features, num_classes)


import math

"""
Unet code courtesy of https://github.com/milesial/Pytorch-UNet codebase.

This NN is modeled after the original Unet paper. 

Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for
Biomedical Image Segmentation.
Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer,
LNCS, Vol.9351: 234--241, 2015


"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, no_classifier=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.no_classifier = no_classifier
        self.output_dim = 64

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        if not self.no_classifier:
            self.outc = OutConv(self.output_dim, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if not self.no_classifier:
            x = self.outc(x)
        return x


def UNet_model(
    n_channels=1,
    n_classes=2,
    bilinear=True,
    no_classifier=False,
):
    """Constructs a UNet model."""
    model = UNet(
        n_channels,
        n_classes,
        bilinear,
        no_classifier,
    )
    return model
