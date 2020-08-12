#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import torch.nn as nn

# from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.contextnet.encoder_layer import Swish

# from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
# from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
# from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.contextnet.encoder_layer import EncoderLayer

# from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
# from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
# from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
#     MultiLayeredConv1d,
# )
# from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
#     PositionwiseFeedForward,  # noqa: H301
# )
# from espnet.nets.pytorch_backend.transformer.repeat import repeat
# from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling

activation_funcs = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": Swish,
}


class Encoder(torch.nn.Module):
    """ContextNet encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        num_blocks=[1],
        filters=[256],
        num_convs=[1],
        kernels=[5],
        strides=[1],
        dilations=[1],
        dropouts=[0.0],
        residuals=[False],
        activation="relu",
        act_last=False,
        se_reduction_ratio=8,
        padding_idx=-1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        in_channels = idim
        activation = activation_funcs[activation]()
        encoder_layers = []
        for i in range(len(num_blocks)):
            for j in range(num_blocks[i]):
                encoder_layers.append(
                    EncoderLayer(
                        in_channels=in_channels,
                        out_channels=filters[i],
                        num_conv=num_convs[i],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        dilation=dilations[i],
                        dropout=dropouts[i],
                        activation=activation,
                        act_last=act_last,
                        residual=residuals[i],
                        se_reduction_ratio=se_reduction_ratio,
                    )
                )
                in_channels = filters[i]

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, xs):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        xs = self.encoder(xs)

        return xs

    # def forward_one_step(self, xs, masks, cache=None):
    #     """Encode input frame.

    #     :param torch.Tensor xs: input tensor
    #     :param torch.Tensor masks: input mask
    #     :param List[torch.Tensor] cache: cache tensors
    #     :return: position embedded tensor, mask and new cache
    #     :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    #     """
    #     if isinstance(self.embed, Conv2dSubsampling):
    #         xs, masks = self.embed(xs, masks)
    #     else:
    #         xs = self.embed(xs)
    #     if cache is None:
    #         cache = [None for _ in range(len(self.encoders))]
    #     new_cache = []
    #     for c, e in zip(cache, self.encoders):
    #         xs, masks = e(xs, masks, cache=c)
    #         new_cache.append(xs)
    #     if self.normalize_before:
    #         xs = self.after_norm(xs)
    #     return xs, masks, new_cache
