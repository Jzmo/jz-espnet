#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging
import torch

from espnet.nets.pytorch_backend.newformer.encoder_layer import (
    EncoderLayerStack,
    EncoderLayerParallel,
)
from espnet.nets.pytorch_backend.newformer.combine_methods import get_dual_projection

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.contextnet.encoder_layer import SELayer
from espnet.nets.pytorch_backend.newformer.convolution import (
    ConformerConvolutionModule,
    LightDepthwiseConvolutionModule,
)
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class Encoder(torch.nn.Module):
    """Dynamicformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_pos_enc_layer_type: encoder positional encoding layer type
    :param str encoder_attn_layer_type: encoder attention layer type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int lightconv_wshare=4: 
    :param int lightconv_dim: 
    :param float lightconv_dropout_rate: 
    :param str lightconv_kernel_length:
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        padding_idx=-1,
        activation_type="swish",
        conv_encoder_type="stack",
        use_pos_enc_conv=False,
        conv_encoder_layer_type="lightconv",
        lightconv_wshare=4,
        lightconv_dim=256,
        lightconv_dropout_rate=0.1,
        conv_usebias=False,
        conv_kernel_length_str="31",
        conv_block_number_str="all",
        ff_block_number_str="all",
        use_se_layer=False,
        se_block_number_str="",
        se_activation=torch.nn.ReLU(),
        se_reduction_ratio=8,
        shuffle_after=False,
        shuffle_block_number_str="",
        dual_type="linear",
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        if conv_encoder_type == "stack":
            encoder_layer = EncoderLayerStack
            attn_block_number = [True for _ in range(num_blocks)]
        elif conv_encoder_type == "alter":
            encoder_layer = EncoderLayerStack
            conv_layers = conv_block_number_str.split("_")
            attn_block_number = [
                False if str(nb) in conv_layers else True
                for nb in range(1, num_blocks + 1)
            ]
        elif conv_encoder_type == "parallel":
            encoder_layer = EncoderLayerParallel
            attn_block_number = [True for _ in range(num_blocks)]
            dual_proj = get_dual_projection(dual_type, attention_dim)
        else:
            raise ValueError("unknown encoder type: " + conv_encoder_type)

        conv_layer_args = None
        if conv_encoder_layer_type == "lightconv":
            logging.info("encoder lightconv layer type = lightconv")
            conv_layer = LightweightConvolution
        elif conv_encoder_layer_type == "lightconv2d":
            logging.info("encoder lightconv layer type = lightconv2d")
            conv_layer = LightweightConvolution2D
        elif conv_encoder_layer_type == "dynamicconv":
            logging.info("encoder lightconv layer type = dynamicconv")
            conv_layer = DynamicConvolution
        elif conv_encoder_layer_type == "dynamicconv2d":
            logging.info("encoder lightconv layer type = dynamicconv2d")
            conv_layer = DynamicConvolution
        elif conv_encoder_layer_type == "conformer_conv1d":
            logging.info("encoder lightconv layer type = conformer1d")
            conv_layer = ConformerConformerConvolutionModule
        elif conv_encoder_layer_type == "lightdepthwise":
            logging.info("encoder lightconv layer type = lightdepthwiseconv1d")
            conv_layer = LightDepthwiseConvolutionModule
        else:
            if conv_encoder_layer_type is not None:
                raise ValueError(
                    "unknown encoder_lightconv_layer: " + conv_encoder_layer_type
                )

        if conv_block_number_str == "all":
            conv_block_number = [True for _ in range(num_blocks)]
        elif "_" in conv_block_number_str or conv_block_number_str.isnumeric():
            layers = conv_block_number_str.split("_")
            conv_block_number = [
                True if str(nb) in layers else False for nb in range(1, num_blocks + 1)
            ]
        else:
            raise ValueError(
                "unknown encoder_conv_layer_number: " + conv_block_number_str
            )
        if ff_block_number_str == "all":
            ff_block_number = [True for _ in range(num_blocks)]
        elif "_" in ff_block_number_str or _block_number_str.isnumeric():
            layers = ff_block_number_str.split("_")
            ff_block_number = [
                True if str(nb) in layers else False for nb in range(1, num_blocks + 1)
            ]
        else:
            raise ValueError("unknown encoder_ff_layer_number:" + ff_block_number_str)

        if use_se_layer:
            if se_block_number_str == "":
                se_block_number = [False for _ in range(num_blocks)]
            else:
                se_layer = SELayer(
                    attention_dim,
                    reduction_ratio=se_reduction_ratio,
                    activation=se_activation,
                )
                layers = se_block_number_str.split("_")
                se_block_number = [
                    True if str(nb) in layers else False
                    for nb in range(1, num_blocks + 1)
                ]
        else:
            se_block_number = [False for _ in range(num_blocks)]

        if shuffle_after:
            if shuffle_block_number_str == "":
                shuffle_block_number = [False for _ in range(num_blocks)]
            else:
                layers = shuffle_block_number_str.split("_")
                shuffle_block_number = [
                    True if str(nb) in layers else False
                    for nb in range(1, num_blocks + 1)
                ]
        else:
            shuffle_block_number = [False for _ in range(num_blocks)]
        self.encoders = repeat(
            num_blocks,
            lambda lnum: encoder_layer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args)
                if attn_block_number[lnum]
                else None,
                positionwise_layer(*positionwise_layer_args)
                if ff_block_number[lnum]
                else None,
                positionwise_layer(*positionwise_layer_args)
                if ff_block_number[lnum] and macaron_style
                else None,
                conv_layer(
                    lightconv_wshare,
                    lightconv_dim,
                    lightconv_dropout_rate,
                    conv_kernel_length_str,
                    lnum,
                    activation=activation,
                    use_bias=conv_usebias,
                )
                if conv_layer is not None and conv_block_number[lnum]
                else None,
                use_pos_enc_conv,
                dual_proj if conv_encoder_type == "parallel" else None,
                se_layer if use_se_layer and se_block_number[lnum] else None,
                shuffle_after and shuffle_block_number[lnum],
                dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
                lnum=lnum,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        xs, masks = self.encoders(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks
