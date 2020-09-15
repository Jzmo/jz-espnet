#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution


class ConformerConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(
        self,
        wshare,  # wshare in lightweight conv, not using here
        channels,
        dropout_rate,
        kernel_size_str,
        lnum,
        activation=nn.ReLU(),
        use_bias=True,
    ):
        """Construct an ConvolutionModule object."""
        super(ConformerConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        self.kernel_size = int(kernel_size_str.split("_")[lnum])
        assert (self.kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.activation = activation

    def forward(self, q, k, v, mask):
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # exchange the temporal dimension and the feature dimension
        x = q.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)


class LightDepthwiseConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(
        self,
        wshare,  # wshare in lightweight conv, not using here
        channels,
        dropout_rate,
        kernel_size_str,
        lnum,
        activation=nn.ReLU(),
        use_bias=True,
    ):
        """Construct an ConvolutionModule object."""
        super(LightDepthwiseConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        self.kernel_size = int(kernel_size_str.split("_")[lnum])
        assert (self.kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=use_bias,
        )
        self.depthwise_conv = LightweightConvolution(
            wshare,
            channels,
            dropout_rate,
            kernel_size_str,
            lnum,
            activation=None,
            use_bias=use_bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=use_bias,
        )
        self.activation = activation

    def forward(self, q, k, v, mask):
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # exchange the temporal dimension and the feature dimension
        x = q
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x.transpose(1, 2), k, v, None)
        x = x.transpose(1, 2)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
