import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class ConcatChannel(torch.nn.Module):
    def __init__(self, size, shuffle=False):
        super(ConcatChannel, self).__init__()
        self.size = size
        self.conv1 = torch.nn.Conv1d(size, size // 2, 1)
        self.conv2 = torch.nn.Conv1d(size, size // 2, 1)
        self.n_group = 8
        self.shuffle = shuffle

    def forward(self, x1, x2):
        B, T, C = x1.size()
        x1 = self.conv1(x1.transpose(1, 2))  # B, C, T, attn output
        x2 = self.conv2(x2.transpose(1, 2))  # B, C, t, conv output
        x = torch.cat((x1, x2), dim=1).transpose(1, 2)  # B, T, C
        if self.shuffle:
            channels_per_group = self.size // self.n_group
            x = x.view(B, T, self.n_group, channels_per_group)
            x = torch.transpose(x, 2, 3).contiguous()
            x = x.view(B, T, C)
        return x


class Attn_CNN(torch.nn.Module):
    def __init__(self, size):
        super(Attn_CNN, self).__init__()
        self.size = size
        self.dropout_rate = 0.1
        self.norm = LayerNorm(size)
        self.normalize_before = True
        self.attn = MultiHeadedAttention(8, size, 0.1)

    def forward(self, x1, x2):
        if self.normalize_before:
            x2 = self.norm(x2)
        residual = x2
        kv = torch.cat((x1, x2))
        x2 = residual + self.attn(x2, kv, kv, None)
        if not self.normalize_before:
            x2 = self.norm(x2)
        return x2


class Attn(torch.nn.Module):
    def __init__(self, size):
        super(Attn, self).__init__()
        self.size = size
        self.dropout_rate = 0.1
        self.norm = LayerNorm(size)
        self.normalize_before = True
        self.attn = MultiHeadedAttention(8, size, 0.1)

    def forward(self, x1, x2):
        if self.normalize_before:
            x1 = self.norm(x1)
        residual = x1
        kv = torch.cat((x1, x2))
        x1 = residual + self.attn(x1, kv, kv, None)
        if not self.normalize_before:
            x1 = self.norm(x1)
        return x1


class Highway(torch.nn.Module):
    def __init__(self, size, f):
        super(Highway, self).__init__()
        self.gate = torch.nn.Linear(size, size)
        self.linear1 = torch.nn.Linear(size, size)
        self.linear2 = torch.nn.Linear(size, size)
        self.f = f

    def forward(self, x1, x2):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
        and ⨀ is element-wise multiplication
        """
        gate = F.sigmoid(self.gate(x1 + x2))
        x = gate * self.f(self.linear1(x1)) + (1 - gate) * self.f(self.linear2(x2))
        return x


class ConcatLinear(torch.nn.Module):
    def __init__(self, size, f):
        super(ConcatLinear, self).__init__()
        self.size = size
        self.f = f
        self.linear = torch.nn.Linear(size * 2, size)

    def forward(self, x1, x2):
        return self.f(self.linear(torch.cat((x1, x2), dim=-1)))


class ConcatAvePooling(torch.nn.Module):
    def __init__(self, size, f):
        super(ConcatAvePooling, self).__init__()
        self.size = size
        self.f = f
        self.pool = torch.nn.AvgPool1d(2, stride=1)

    def forward(self, x1, x2):
        return self.f(self.pool(torch.cat((x1, x2), dim=-1)))


class ConcatMaxPooling(torch.nn.Module):
    def __init__(self, size, f):
        super(ConcatMaxPooling, self).__init__()
        self.size = size
        self.f = f
        self.pool = torch.nn.MaxPool1d(2, stride=1)

    def forward(self, x1, x2):
        return self.f(self.pool(torch.cat((x1, x2), dim=-1)))
