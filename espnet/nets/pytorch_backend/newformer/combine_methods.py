import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class _ConcatChannel(torch.nn.Module):
    def __init__(self, size):
        super(_ConcatChannel, self).__init__()
        self.size = size
        self.conv_k1 = torch.nn.Conv1d(size * 2, size, 1)
        self.n_group = 8

    def forward(self, x1, x2):
        B, T, C = x1.size()
        x = self.conv_k1(
            torch.cat((x1, x2), dim=-1).transpose(1, 2)
        )  # B, C, T, attn output

        return x.transpose(1, 2)


class _Self_Mix_Attn(torch.nn.Module):
    def __init__(self, size):
        super(_Self_Mix_Attn, self).__init__()
        self.size = size
        self.dropout_rate = 0.1
        self.norm = LayerNorm(size)
        self.attn = MultiHeadedAttention(8, size, 0.1)

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        residual = x1
        kv = torch.cat((x1, x2))
        x1 = residual + self.attn(x1, kv, kv, None)

        return x1


class _GLU(torch.nn.Module):
    def __init__(self, size):
        super(_GLU, self).__init__()
        self.act = torch.nn.GLU()
        self.linear1 = torch.nn.Linear(size * 2, size * 2)
        self.linear2 = torch.nn.Linear(size, size)

    def forward(self, x1, x2):
        x = self.linear1(torch.cat((x1, x2), dim=-1))
        # GLU activation
        x = self.act(x)
        x = self.linear2(x)

        return x


class _ConcatLinear(torch.nn.Module):
    def __init__(self, size, f=torch.nn.ReLU):
        super(_ConcatLinear, self).__init__()
        self.size = size
        self.f = f
        self.linear = torch.nn.Linear(size * 2, size)

    def forward(self, x1, x2):
        x = self.f(self.linear(torch.cat((x1, x2), dim=-1)))

        return x


class _ConcatAvePooling(torch.nn.Module):
    def __init__(self, size, f=torch.nn.ReLU):
        super(_ConcatAvePooling, self).__init__()
        self.size = size
        self.f = f
        self.pool = torch.nn.AvgPool1d(2, stride=1)

    def forward(self, x1, x2):
        x = self.f(self.pool(torch.cat((x1, x2), dim=-1)))

        return x


class _ConcatMaxPooling(torch.nn.Module):
    def __init__(self, size, f=torch.nn.ReLU):
        super(_ConcatMaxPooling, self).__init__()
        self.size = size
        self.f = f
        self.pool = torch.nn.MaxPool1d(2, stride=1)

    def forward(self, x1, x2):
        x = self.f(self.pool(torch.cat((x1, x2), dim=-1)))

        return x


def get_dual_projection(proj, size):
    """Return projection function."""
    projection_funcs = {
        "concat_ch": _ConcatChannel(size),
        "self_mix_attn": _Self_Mix_Attn(size),
        "glu": _GLU(size),
        "concat_linear": _ConcatLinear(size),
        "concat_avgpool": _ConcatAvePooling(size),
        "concat_maxpool": _ConcatMaxPooling(size),
    }

    return projection_funcs[proj]
