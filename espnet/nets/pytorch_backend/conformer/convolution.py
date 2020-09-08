#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(
        self,
        channels,
        kernel_size,
        activation=nn.ReLU(),
        use_lightweight=False,
        bias=True,
    ):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        if not cnn_use_lightweight:
            self.depthwise_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=channels,
                bias=bias,
            )
        elif cnn_use_lightweight:
            if cnn_lightconv_layer_type == "lightconv":
                logging.info("encoder lightconv layer type = lightconv")
                lightconv_layer = LightweightConvolution
            elif cnn_lightconv_layer_type == "lightconv2d":
                logging.info("encoder lightconv layer type = lightconv2d")
                lightconv_layer = LightweightConvolution2D
            elif cnn_lightconv_layer_type == "dynamicconv":
                logging.info("encoder lightconv layer type = dynamicconv")
                lightconv_layer = DynamicConvolution
            elif cnn_lightconv_layer_type == "dynamicconv2d":
                logging.info("encoder lightconv layer type = dynamicconv2d")
                lightconv_layer = DynamicConvolution
            else:
                if lightconv_layer is not None:
                    raise ValueError(
                        "unknown encoder_lightconv_layer: " + lightconv_layer_type
                    )
            self.depthwise_conv = lightconv_layer(
                lightweight_wshare,
                channels,
                0.1,
                kernel_size,
                lnum,
                use_kernel_mask=False,
                use_bias=bias,
            )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
