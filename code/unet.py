# MIT License

# Original work Copyright (c) 2018 Joris (https://github.com/jvanvugt/pytorch-unet)
# Modified work Copyright (C) 2022 Canon Medical Systems Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from math import sqrt
from abc import abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from swish import CustomSwish
from ws_conv import WNConv2d

from monai.networks.nets import SwinUNETR


def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """




class Dubble_Unet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=False,
            norm="group",
            up_mode='upconv'):
        super(Dubble_Unet, self).__init__()
        self.Unet1 = UNet(in_channels, n_classes, norm=norm, up_mode=up_mode, depth=depth, wf=wf, padding=padding)
        self.Unet2 = UNet(in_channels, n_classes, norm=norm, up_mode=up_mode, depth=depth, wf=wf, padding=padding)

    def forward(self, x1, x2):
        y1 = self.Unet1(x1)
        y2 = self.Unet2(x2)
        return y1, y2

    def forward_with_last(self, x1, x2):
        y1, y1_l = self.Unet1.forward_with_last(x1)
        y2, y2_l = self.Unet2.forward_with_last(x2)
        return y1, y2, y1_l, y2_l

    def forward_with_inner(self, x1, x2):
        y1, y1_l = self.Unet1.forward_with_inner(x1)
        y2, y2_l = self.Unet2.forward_with_inner(x2)
        return y1, y2, y1_l, y2_l

    def forward_with_all(self, x1, x2):
        y1, y1_l, y1_i = self.Unet1.forward_with_all(x1)
        y2, y2_l, y2_i = self.Unet2.forward_with_all(x2)
        return y1, y2, y1_l, y2_l, y1_i, y2_i





class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=False,
            norm="group",
            up_mode='upconv'):
        """
        A modified U-Net implementation [1].

        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward_down(self, x):

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)

        return x, blocks

    def forward_up_without_last(self, x, blocks):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)

        return x

    def forward_without_last(self, x):

        x, blocks = self.forward_down(x)
        inner_f = x
        x = self.forward_up_without_last(x, blocks)
        return x, inner_f

    def forward(self, x):
        x, inner_f = self.get_features(x)
        return self.last(x)

    def forward_with_last(self, x):
        x, inner_f = self.get_features(x)
        return self.last(x), x

    def forward_with_all(self, x):
        x, inner_f = self.get_features(x)
        return self.last(x), x, inner_f

    def forward_with_inner(self, x):
        x, inner_f = self.get_features(x)
        return self.last(x), inner_f
    def get_features(self, x):
        return self.forward_without_last(x)

class DUNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=False,
            norm="group",
            up_mode='upconv'):
        """
        A modified U-Net implementation [1].

        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(DUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.first = nn.Conv2d(in_channels, 2 ** (wf -2), kernel_size=1)
        prev_channels = 2 ** wf
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes * 4, kernel_size=1)

    def forward_down(self, x):

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)

        return x, blocks

    def forward_up_without_last(self, x, blocks):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)

        return x

    def forward_without_last(self, x):
        x, blocks = self.forward_down(x)
        x = self.forward_up_without_last(x, blocks)
        return x

    def forward(self, x):
        mods = x.transpose(1, 0)
        x = []
        for mod in mods:
            x.append(self.first(mod.unsqueeze(1)))
        x = torch.cat(x, dim=1)
        x = self.get_features(x)
        return self.last(x)

    def get_first(self, x):
        mods = x.transpose(1, 0)
        x = []
        for mod in mods:
            x.append(self.first(mod.unsqueeze(1)))
        return x

    def forward_start_with_first(self, x):
        x = self.get_features(x)
        return self.last(x)

    def get_features(self, x):
        return self.forward_without_last(x)

    def forward_with_last(self, x):
        x = self.get_features(x)
        return self.last(x), x

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == '__main__':
    net = condition_UNet(in_channels=4, depth=4, wf=7, padding=True)
    a = torch.randn(1, 4, 128, 128)
    c = torch.randn(1, 1, 1280)
    b = net(a)
    print(b.size())
