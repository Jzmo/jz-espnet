#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

import torch
from torch import nn
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(self, channels, kernel_size, k2=None, bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.depthwise_conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        k2 = (kernel_size + 1) // 4 - 1
        self.depthwise_conv2 = nn.Conv1d(
            channels,
            channels,
            k2,
            stride=1,
            padding=(k2 - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.pointwise_conv3 = nn.Conv1d(
            channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.activation = Swish()

    def forward(self, x):
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # exchange the temporal dimension and the feature dimension
        T = x.size(1)
        C = x.size(2)
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x1 = self.depthwise_conv1(x)
        x2 = self.depthwise_conv2(x)
        x1 = self.activation(self.norm1(x1))
        x2 = self.activation(self.norm2(x2))

        x1 = self.pointwise_conv2(x1)
        x2 = self.pointwise_conv3(x2)
        x = torch.cat([x1, x2], dim=1)  # (batch, channel, time)

        return x.transpose(1, 2)  # (batch, time, channel)


class Swish(nn.Module):
    """Construct an Swish function object."""

    def forward(self, x):
        """Return an Swich activation function."""
        return x * torch.sigmoid(x)
