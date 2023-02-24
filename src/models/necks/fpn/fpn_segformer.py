import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.necks.fpn.fpn import ConvBnRelu

class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerFPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.linear_c4 = MLP(int(in_channels[3]), embed_dim=inner_channels)
        self.linear_c3 = MLP(int(in_channels[2]), embed_dim=inner_channels)
        self.linear_c2 = MLP(int(in_channels[1]), embed_dim=inner_channels)
        self.linear_c1 = MLP(int(in_channels[0]), embed_dim=inner_channels)

        self.dropout = nn.Dropout2d(0.1)
        self.linear_fuse = ConvBnRelu(
            in_channels=inner_channels*4,
            out_channels=inner_channels,
            kernel_size=1,
            bias=False)

        self.linear_pred = nn.Conv2d(
            inner_channels, self.conv_out, kernel_size=1)
        self.out_channels = self.conv_out

    def forward(self, x):
        c1, c2, c3, c4 = x

        c1_shape = c1.size()
        c2_shape = c2.size()
        c3_shape = c3.size()
        c4_shape = c4.size()

        all_hidden_states = ()

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape([c4_shape[0], -1, c4_shape[2], c4_shape[3]])

        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=False)
        all_hidden_states += (_c4,)

        _c3 = self.linear_c3(c3).permute([0, 2, 1]).reshape(
            [c3_shape[0], -1, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=False)
        all_hidden_states += (_c3,)

        _c2 = self.linear_c2(c2).permute([0, 2, 1]).reshape(
            [c2_shape[0], -1, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=False)
        all_hidden_states += (_c2,)


        _c1 = self.linear_c1(c1).permute([0, 2, 1]).reshape(
            [c1_shape[0], -1, c1_shape[2], c1_shape[3]])
        all_hidden_states += (_c1,)

        _c = self.linear_fuse(torch.cat(all_hidden_states[::-1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return logit