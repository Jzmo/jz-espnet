#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
import numpy as np

from torch import nn
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


def write_to(filename, tensor):
    with open(filename, "ab") as f:
        # Append 'hello' at the end of file
        if len(tensor.size()) > 2:
            for tensor_slice in tensor:
                np.savetxt(f, tensor_slice.numpy())
        else:
            np.savetxt(f, tensor.numpy())
    f.close()


class EncoderLayerStack(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
        RelPositionMultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward
    for macaron style
    PositionwiseFeedForward feed_forward:
    feed forward module
    :param espnet.nets.pytorch_backend.conformer.convolution.
        ConvolutionModule feed_foreard:
        feed forward module
    :param espent.nets.pytorch_backend.transformer.dynamic_conv.
    :param espent.nets.pytorch_backend.transformer.dynamic_conv2d.
    :param espent.nets.pytorch_backend.transformer.light_conv.
    :param espent.nets.pytorch_backend.transformer.light_conv2d.
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv,
        use_pos_enc_conv,
        dual_proj,
        se_layer,
        shuffle_after,
        dropout_rate,
        ff_scale=1.0,
        normalize_before=True,
        concat_after=False,
        lnum=None,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerStack, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv = conv
        self.use_pos_enc_conv = use_pos_enc_conv
        self.se_layer = se_layer
        self.shuffle_after = shuffle_after
        self.ff_scale = ff_scale
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        if self.conv is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.lnum = lnum

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features, w/o pos_emb
        tuple((batch, max_time_in, size), (1, max_time_in, size))
        or (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # write_to("feature_map/before_ffm.lnum{}".format(self.lnum), x)

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # write_to("feature_map/after_ffm.lnum{}".format(self.lnum), x)

        # multi-headed self-attention module
        if self.self_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_mha(x)
            if cache is None:
                x_q = x
            else:
                assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
                x_q = x[:, -1:, :]
                residual = residual[:, -1:, :]
                mask = None if mask is None else mask[:, -1:, :]

            if pos_emb is not None:
                x_att = self.self_attn(x_q, x, x, pos_emb, mask)
            else:
                x_att = self.self_attn(x_q, x, x, mask)

            if self.concat_after:
                x_concat = torch.cat((x, x_att), dim=-1)
                x = residual + self.concat_linear(x_concat)
            else:
                x = residual + self.dropout(x_att)
            if not self.normalize_before:
                x = self.norm_mha(x)
        # write_to("feature_map/after_attn.lnum{}".format(self.lnum), x)
        # light convolution module
        if self.conv is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            if self.use_pos_enc_conv:
                x = x + pos_emb
            x = self.dropout(self.conv(x, x, x, None))
            x = residual + x
            if not self.normalize_before:
                x = self.norm_conv(x)  # B, T, C
        # write_to("feature_map/after_conv.lnum{}".format(self.lnum), x)
        if self.se_layer is not None:
            residual = x
            x = self.se_layer(x.transpose(1, 2))  # B, T, C -> B, C, T
            x = residual + x.transpose(1, 2)  # B, C, T -> B, T, C

        if self.shuffle_after:
            B, T, C = x.size()
            x = x.view(B, T, C // 2, 2)
            x = torch.transpose(x, 2, 3).contiguous()
            x = x.view(B, T, C)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)
        # write_to("feature_map/after_ff.lnum{}".format(self.lnum), x)
        if self.conv is not None:
            x = self.norm_final(x)
        # write_to("feature_map/after_finalnorm.lnum{}".format(self.lnum), x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class EncoderLayerParallel(nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv,
        use_pos_enc_conv,
        dual_proj,
        se_layer,
        shuffle_after,
        dropout_rate,
        ff_scale=1.0,
        normalize_before=True,
        concat_after=False,
    ):
        super(EncoderLayerParallel, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv = conv
        self.use_pos_enc_conv = use_pos_enc_conv
        self.dual_proj = dual_proj
        self.se_layer = se_layer
        self.shuffle_after = shuffle_after
        self.ff_scale = ff_scale
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        if self.conv is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features, w/o pos_emb
        tuple((batch, max_time_in, size), (1, max_time_in, size))
        or (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        # multi-headed self-attention module
        x_att = None
        if self.self_attn is not None:
            if cache is None:
                x_q = x
            else:
                assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
                x_q = x[:, -1:, :]
                residual = residual[:, -1:, :]
                mask = None if mask is None else mask[:, -1:, :]

            if pos_emb is not None:
                x_att = self.self_attn(x_q, x, x, pos_emb, mask)
            else:
                x_att = self.self_attn(x_q, x, x, mask)

            if self.concat_after:
                x_concat = torch.cat((x_q, x_att), dim=-1)
                x_att = residual + self.concat_linear(x_concat)
            else:
                x_att = residual + self.dropout(x_att)
            if not self.normalize_before:
                x_att = self.norm_mha(x_att)

        # light convolution module
        x_conv = None
        if self.conv is not None:
            if self.use_pos_enc_conv:
                x_conv = x + pos_emb
                x_conv = residual + self.dropout(self.conv(x_conv, None, None, None))
            else:
                x_conv = residual + self.dropout(self.conv(x, None, None, None))
            if not self.normalize_before:
                x_conv = self.norm_conv(x_conv)  # B, T, C

        if x_att is not None and x_conv is not None:
            x = self.dual_proj(x_att, x_conv)
        elif x_att is not None and x_conv is None:
            x = x_att
        elif x_att is None and x_conv is not None:
            x = x_conv
        else:
            raise ValueError("neither self attention or convolution module exist")

        if self.se_layer is not None:
            residual = x
            x = self.se_layer(x.transpose(1, 2))  # B, T, C -> B, C, T
            x = residual + x.transpose(1, 2)  # B, C, T -> B, T, C

        if self.shuffle_after:
            B, T, C = x.size()
            x = x.view(B, T, C // 2, 2)
            x = torch.transpose(x, 2, 3).contiguous()
            x = x.view(B, T, C)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask