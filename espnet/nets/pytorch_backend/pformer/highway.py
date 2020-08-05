import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = [nn.Linear(size, size) for _ in range(2)]
        self.gate = nn.Linear(size, size)
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
        x1 = self.f(self.nonlinear[0](x1))
        x2 = self.nonlinear[1](x2)

        x = gate * x1 + (1 - gate) * x2

        return x
