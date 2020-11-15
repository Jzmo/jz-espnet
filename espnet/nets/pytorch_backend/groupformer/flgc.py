import torch
import torch.nn as nn


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(context, probs):
        binarized = (probs == torch.max(probs, dim=1, keepdim=True)[0]).float()
        context.save_for_backward(binarized)
        return binarized

    @staticmethod
    def backward(context, gradient_output):
        (binarized,) = context.saved_tensors
        gradient_output[binarized == 0] = 0
        return gradient_output


class Flgc1d(nn.Module):
    def __init__(
        self,
        groups,
        conv_dim,
        dropout_rate,
        kernel_size,
        layer_num,
        use_bias=True,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super().__init__()
        self.padding_size = int(kernel_size / 2)
        self.dropout_rate = dropout_rate
        self.in_channels_in_group_assignment_map = nn.Parameter(
            torch.Tensor(conv_dim, groups)
        )
        nn.init.normal_(self.in_channels_in_group_assignment_map)
        self.out_channels_in_group_assignment_map = nn.Parameter(
            torch.Tensor(conv_dim, groups)
        )
        nn.init.normal_(self.out_channels_in_group_assignment_map)
        self.conv = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size,
            stride,
            self.padding_size,
            dilation,
            1,
            use_bias,
        )
        self.binarize = Binarize.apply

    def forward(self, query, key, value, mask):
        x = query
        B, T, C = x.size()
        x = x.transpose(1, 2).contiguous().view(-1, C, T)  # B x C x T
        channel_map = torch.mm(
            self.binarize(
                torch.softmax(self.out_channels_in_group_assignment_map, dim=1)
            ),
            torch.t(
                self.binarize(
                    torch.softmax(self.in_channels_in_group_assignment_map, dim=1)
                )
            ),
        )
        weight = torch.nn.functional.dropout(
            self.conv.weight, self.dropout_rate, training=self.training
        )
        x = torch.nn.functional.conv1d(
            x,
            weight * channel_map[:, :, None],
            self.conv.bias,
            self.conv.stride,
            self.padding_size,
            self.conv.dilation,
        )
        x = x.transpose(1, 2)  # B x T x C
        if mask is not None:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)
        return x
