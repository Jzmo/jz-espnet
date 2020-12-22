#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import numpy
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)


class RelPositionMixMultiHeadAttentionDSConvolution(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head, n_feat,
                 dropout_rate,
                 ac_attn_head_ratio,
                 ac_cnn_module_kernel,
                 ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)

        input_n_feat = n_feat

        self.d_k = input_n_feat // n_head
        new_n_head = int(n_head // ac_attn_head_ratio)
        if new_n_head < 1:
            ac_attn_head_ratio = n_head
            new_n_head = 1
        self.h = new_n_head
        n_feat = self.d_k * self.h
        self.linear_q = nn.Linear(input_n_feat, n_feat)
        self.linear_k = nn.Linear(input_n_feat, n_feat)
        self.linear_v = nn.Linear(input_n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, input_n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.ac_cnn_module_kernel = ac_cnn_module_kernel
        self.pointwise_conv1 = nn.Conv1d(
            input_n_feat,
            2 * n_feat,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.depthwise_conv = nn.Conv1d(
            n_feat,
            n_feat,
            ac_cnn_module_kernel,
            stride=1,
            padding=(ac_cnn_module_kernel-1) // 2,
            groups=n_feat,
        )
        self.pointwise_conv2 = nn.Conv1d(
            n_feat,
            n_feat,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.linear_conv = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(
            2 * n_feat, input_n_feat)
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(
            input_n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(
            torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(
            torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(
            self.pos_bias_u)
        torch.nn.init.xavier_uniform_(
            self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of the matrix.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros(
            (*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat(
            [zero_pad, x], dim=-1)

        x_padded = x_padded.view(
            *x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :,
                     1:].view_as(x)

        if zero_triu:
            ones = torch.ones(
                (x.size(2), x.size(3)))
            x = x * \
                torch.tril(ones, x.size(
                    3) - x.size(2))[None, None, :, :]

        return x

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(
                0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(
                    0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(
                mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            # (batch, head, time1, time2)
            self.attn = torch.softmax(
                scores, dim=-1)

        p_attn = self.dropout(self.attn)
        # (batch, head, time1, d_k)
        x = torch.matmul(p_attn, value)

        # (batch, head, time1, d_k)
        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(
            query, key, value)
        # (batch, time1, head, d_k)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(
            n_batch_pos, -1, self.h, self.d_k)
        # (batch, head, time1, d_k)
        p = p.transpose(1, 2)

        # (batch, head, time1, d_k)
        q_with_bias_u = (
            q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (
            q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(
            q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(
            q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(
            matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        attn_context = self.forward_attention(
            v, scores, mask).transpose(-1, -2)  # (batch, head, d_k, time)

        # span dynamic convolution
        # (#batch, channel, time2)
        k_conv = key.transpose(1, 2)
        k_conv = self.pointwise_conv1(k_conv)
        k_conv = self.depthwise_conv(nn.functional.glu(k_conv, dim=1))
        k_conv = self.pointwise_conv2(k_conv).transpose(1, 2)

        n_batch, T, _ = query.size()
        q_conv = q.contiguous().view(n_batch, -1, self.h *
                                     self.d_k)  # (#batch, time, channel)
        kernel_weight = q_conv.mul(
            k_conv)  # (#batch, time, kH)
        kernel_weight = self.dropout(kernel_weight)
        kernel_weight = kernel_weight.view(
            n_batch, -1, self.h, self.d_k).transpose(1, 2).contiguous()
        weight_new = torch.zeros(
            n_batch * self.h * T * (T + self.d_k - 1), dtype=kernel_weight.dtype)
        weight_new = weight_new.view(
            n_batch, self.h, T, T + self.d_k - 1).fill_(float("-inf")).to(q_conv.device)
        weight_new.as_strided((n_batch, self.h, T, self.d_k), ((
            T + self.d_k-1) * T * self.h, (T + self.d_k - 1) * T, T + self.d_k, 1)
        ).copy_(kernel_weight)
        weight_new = weight_new.narrow(-1, int((self.d_k-1)/2), T)
        kernel_weight = torch.softmax(
            weight_new, dim=-1).view(n_batch * self.h, T, T)  # (batch*head, time, time)

        conv_v = v.contiguous().view(n_batch * self.h, T, self.d_k)

        conv_context = torch.bmm(kernel_weight, conv_v)  # (batch, time, d_k)
        conv_context = conv_context.view(
            n_batch, self.h, T, self.d_k).transpose(-1, -2)

        mix_context = (torch.cat([attn_context,
                                  conv_context.contiguous()
                                  .view(n_batch, self.h, self.d_k, -1)], dim=1)
                       .contiguous()
                       .view(n_batch, self.h * self.d_k, -1))  # (batch, channel, time1)
        # (#batch, time1, channel)
        mix_context = self.linear_out(
            mix_context.contiguous().view(n_batch, 2 * self.h * self.d_k, T).transpose(-1, -2))

        return mix_context
