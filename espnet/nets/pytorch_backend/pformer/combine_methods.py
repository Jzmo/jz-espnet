import torch
import torch.nn.functional as F


class Highway(torch.nn.Module):
    def __init__(self, size, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.linear = [torch.nn.Linear(size, size) for _ in range(2)]
        self.gate = torch.nn.Linear(size, size)
        self.f = f

    def forward(self, x1, x2):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
        and ⨀ is element-wise multiplication
        """
        gate = F.sigmoid(self.gate[layer](x1 + x2))
        x1 = self.f(self.linear[0](x1))
        x2 = self.f(self.linear[1](x2))
        x = gate * x1 + (1 - gate) * x2
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
        super(ConcatPooling, self).__init__()
        self.size = size
        self.f = f
        self.pool = nn.AvgPool1d(2, stride=1)

    def forward(self, x1, x2):
        return self.f(self.linear(torch.cat((x1, x2), dim=-1)))


class ConcatMaxPooling(torch.nn.Module):
    def __init__(self, size, f):
        super(ConcatPooling, self).__init__()
        self.size = size
        self.f = f
        self.pool = nn.MaxPool1d(2, stride=1)

    def forward(self, x1, x2):
        return self.f(self.linear(torch.cat((x1, x2), dim=-1)))
