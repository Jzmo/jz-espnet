#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Conved Attention layer definition."""

import math

import numpy
import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.mcformer.dynamic_conv import DynamicConvolution


class MultiConvedAttention(nn.Module):
    """Multi-Conved Attention layer.

    :param int n_conv: the number of conv s
    :param int n_feat: the number of features
    :param int n_kernel: the number of kernels in convolution
    :param str kernel_size_str: the kernel size for multi convs: "1_3_5_7"
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, n_kernel, kernel_size_str, dropout_rate):
        """Construct an MultiConvedAttention object."""
        super(MultiConvedAttention, self).__init__()
        assert n_feat % n_head == 0
        assert n_feat % n_kernel == 0
        # We assume d_v always equals d_k
        self.n_feat = n_feat
        self.d_k = n_feat // n_head
        self.d_c = n_feat // n_kernel
        self.h = n_head
        self.n_kernel = n_kernel
        self.kernel_size_str = [kernel for kernel in kernel_size_str.split("_")]
        self.conv_k = repeat(
            self.n_kernel,
            lambda knum: DynamicConvolution(
                1, n_feat, self.d_c, 0.1, self.kernel_size_str[knum], 0
            ),
        )
        self.conv_v = repeat(
            self.n_kernel,
            lambda knum: DynamicConvolution(
                1, n_feat, self.d_c, 0.1, self.kernel_size_str[knum], 0
            ),
        )
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_conv(self, query, key, value):
        """
        Multi-kernel convolution
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)

        """
        T = key.size(1)
        q = query
        k = [self.conv_k[n](key, None, None, None) for n in range(self.n_kernel)]
        v = [self.conv_v[n](value, None, None, None) for n in range(self.n_kernel)]
        k = key + torch.stack(k).permute(1, 2, 0, 3).contiguous().view(
            -1, T, self.n_feat
        )
        v = value + torch.stack(v).permute(1, 2, 0, 3).contiguous().view(
            -1, T, self.n_feat
        )
        return q, k, v

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :return torch.Tensor transformed query, key and value

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        :param torch.Tensor value: (batch, head, time2, size)
        :param torch.Tensor scores: (batch, head, time1, time2)
        :param torch.Tensor mask: (batch, 1, time2) or (batch, time1, time2)
        :return torch.Tensor transformed `value` (batch, time1, d_model)
            weighted by the attention score (batch, time1, time2)

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, 1, time2) or (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output (batch, time1, d_model)
        """
        q, k, v = self.forward_conv(query, key, value)
        q, k, v = self.forward_qkv(q, k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiConvedAttention(MultiConvedAttention):
    """Multi-Head Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, n_kernel, kernel_size_str, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, n_kernel, kernel_size_str, dropout_rate)
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.

        :param torch.Tensor x: (batch, time, size)
        :param bool zero_triu: return the lower triangular part of the matrix
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor pos_emb: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attention output  (batch, time1, d_model)
        """
        q, k, v = self.forward_conv(query, key, value)
        q, k, v = self.forward_qkv(q, k, v)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
