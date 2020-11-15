#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
import torch.nn as nn


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        # padding = dilation * (kernel_size -1) / 2
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channels, reduction_ratio, activation):
        super(SELayer, self).__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()  # [B, C, T]
        y = self.pool(x).view(b, c)  # [B, C, 1] -> [B, C]
        y = self.fc(y).view(b, c, 1)  # [B, C, 1]

        return x * y.expand_as(x)


class DepthwiseConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(DepthwiseConvLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=bias,
            ),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=bias,
            ),
        )
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

    def forward(self, x):
        y = self.norm(self.conv(x))

        return y


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_conv=5,
        kernel_size=5,
        stride=1,
        dilation=1,
        dropout=0.0,
        activation=nn.ReLU(),
        residual=True,
        act_last=False,
        se_reduction_ratio=8,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()

        # ensure "SAME" padding
        padding_val = get_same_padding(kernel_size, stride, dilation)
        in_channels_loop = in_channels
        self.convs = nn.ModuleList()

        # only use stride for the last convolution layer
        for i in range(num_conv - 1):
            self.convs.append(
                DepthwiseConvLayer(
                    in_channels_loop,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_val,
                    dilation=dilation,
                )
            )

            self.convs.append(activation)
            self.convs.append(nn.Dropout(p=dropout))
            in_channels_loop = out_channels

        self.convs.append(
            DepthwiseConvLayer(
                in_channels_loop,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_val,
                dilation=dilation,
            )
        )

        if act_last:
            self.convs.append(activation)
            self.convs.append(nn.Dropout(p=dropout))

        self.convs.append(
            SELayer(
                out_channels, reduction_ratio=se_reduction_ratio, activation=activation
            )
        )

        if residual:
            self.res = nn.ModuleList()
            self.res.append(
                DepthwiseConvLayer(
                    in_channels, out_channels, kernel_size=1, stride=stride
                )
            )
        else:
            self.res = None

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x

        for i, l in enumerate(self.convs):
            x = l(x)

        if self.res is not None:
            for i, l in enumerate(self.res):
                residual = l(residual)
            x = x + residual

        x = self.dropout(self.activation(x))

        return x
