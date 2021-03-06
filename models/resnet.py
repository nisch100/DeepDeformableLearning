<<<<<<< HEAD
import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    """Convolution with weight standardization."""

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    def __init__(self, in_channels, features, stride=1):
        super().__init__()

        self.conv1 = StdConv2d(in_channels, features, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(32, features, eps=1e-6)
        self.conv2 = StdConv2d(features, features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, features, eps=1e-6)
        self.conv3 = StdConv2d(features, features * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(32, features * 4, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != features * 4:
            # Projection also with pre-activation according to paper.
            self.downsample = StdConv2d(in_channels, features * 4, kernel_size=1, stride=stride, padding=0, bias=False)
            self.gn_proj = nn.GroupNorm(features * 4, features * 4)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetStage(nn.Module):
    """A ResNet stage."""

    def __init__(self, in_channels: int, nout: int, first_stride: int, block_size: int):
        super().__init__()

        units = OrderedDict()
        units['unit1'] = ResidualUnit(in_channels=in_channels, features=nout, stride=first_stride)
        for i in range(1, block_size):
            units[f'unit{i + 1}'] = ResidualUnit(in_channels=nout*4, features=nout, stride=1)
        self.units = nn.Sequential(units)

    def forward(self, x):
        return self.units(x)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()

        self.width = int(64 * width_factor)

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, self.width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, self.width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        blocks = OrderedDict()
        blocks['block1'] = ResNetStage(
            in_channels=self.width, 
            nout=self.width, 
            first_stride=1, 
            block_size=block_units[0]
        )
        for i, block_size in enumerate(block_units[1:], 1):
            blocks[f'block{i + 1}'] = ResNetStage(
                in_channels=(self.width * 2**(i + 1)), 
                nout=(self.width * 2**i), 
                first_stride=2, 
                block_size=block_size
            )
        self.body = nn.Sequential(blocks)

    def forward(self, x):
        # Image size, need to determine correct padding size
        in_size = x.size()[2]

        # Skip connections for further layers (decoder)
        skip_connections = []

        # Root
        x = self.root(x)
        skip_connections.append(x)
        x = self.maxpool(x)

        # ResNet blocks
        for i, block in enumerate(self.body):
            x = block(x)
            # Add skip connections for every block except the last one
            if i != len(self.body) - 1:
                pad = int(in_size / 4 / 2**i) - x.size()[2]
                assert pad < 3 and pad >= 0
                skip_connection = F.pad(input=x, pad=(0, pad, 0, pad), mode='constant', value=0)
                skip_connections.append(skip_connection)

=======
import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    """Convolution with weight standardization."""

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    def __init__(self, in_channels, features, stride=1):
        super().__init__()

        self.conv1 = StdConv2d(in_channels, features, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(32, features, eps=1e-6)
        self.conv2 = StdConv2d(features, features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, features, eps=1e-6)
        self.conv3 = StdConv2d(features, features * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(32, features * 4, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != features * 4:
            # Projection also with pre-activation according to paper.
            self.downsample = StdConv2d(in_channels, features * 4, kernel_size=1, stride=stride, padding=0, bias=False)
            self.gn_proj = nn.GroupNorm(features * 4, features * 4)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetStage(nn.Module):
    """A ResNet stage."""

    def __init__(self, in_channels: int, nout: int, first_stride: int, block_size: int):
        super().__init__()

        units = OrderedDict()
        units['unit1'] = ResidualUnit(in_channels=in_channels, features=nout, stride=first_stride)
        for i in range(1, block_size):
            units[f'unit{i + 1}'] = ResidualUnit(in_channels=nout*4, features=nout, stride=1)
        self.units = nn.Sequential(units)

    def forward(self, x):
        return self.units(x)


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()

        self.width = int(64 * width_factor)

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, self.width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, self.width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        blocks = OrderedDict()
        blocks['block1'] = ResNetStage(
            in_channels=self.width, 
            nout=self.width, 
            first_stride=1, 
            block_size=block_units[0]
        )
        for i, block_size in enumerate(block_units[1:], 1):
            blocks[f'block{i + 1}'] = ResNetStage(
                in_channels=(self.width * 2**(i + 1)), 
                nout=(self.width * 2**i), 
                first_stride=2, 
                block_size=block_size
            )
        self.body = nn.Sequential(blocks)

    def forward(self, x):
        # Image size, need to determine correct padding size
        in_size = x.size()[2]

        # Skip connections for further layers (decoder)
        skip_connections = []

        # Root
        x = self.root(x)
        skip_connections.append(x)
        x = self.maxpool(x)

        # ResNet blocks
        for i, block in enumerate(self.body):
            x = block(x)
            # Add skip connections for every block except the last one
            if i != len(self.body) - 1:
                pad = int(in_size / 4 / 2**i) - x.size()[2]
                assert pad < 3 and pad >= 0
                skip_connection = F.pad(input=x, pad=(0, pad, 0, pad), mode='constant', value=0)
                skip_connections.append(skip_connection)

>>>>>>> 50b28318ad5ddd1b9cb087ed827ba0a0255cb753
        return x, skip_connections[::-1]