from torch.hub import load_state_dict_from_url
import torch.nn as nn
from src.models.backbones.convmlp_utils import ConvTokenizer, ConvStage, BasicStage


__all__ = ['ConvMLP', 'convmlp_s', 'convmlp_m', 'convmlp_l']


model_urls = {
    'convmlp_s': 'http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_s_imagenet.pth',
    'convmlp_m': 'http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_m_imagenet.pth',
    'convmlp_l': 'http://ix.cs.uoregon.edu/~alih/conv-mlp/checkpoints/convmlp_l_imagenet.pth',
}


class ConvMLP(nn.Module):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 classifier_head=True,
                 num_classes=1000,
                 *args, **kwargs):
        super(ConvMLP, self).__init__()
        assert len(blocks) == len(dims) == len(mlp_ratios), \
            f"blocks, dims and mlp_ratios must agree in size, {len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed."

        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(n_conv_blocks,
                                     embedding_dim_in=channels,
                                     hidden_dim=dims[0],
                                     embedding_dim_out=dims[0])

        self.stages = nn.ModuleList()
        for i in range(0, len(blocks)):
            stage = BasicStage(num_blocks=blocks[i],
                               embedding_dims=dims[i:i + 2],
                               mlp_ratio=mlp_ratios[i],
                               stochastic_depth_rate=0.1,
                               downsample=(i + 1 < len(blocks)))
            self.stages.append(stage)
        if classifier_head:
            self.norm = nn.LayerNorm(dims[-1])
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.head = None
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        x = x.permute(0, 2, 3, 1)
        for stage in self.stages:
            x = stage(x)
        if self.head is None:
            return x
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


def _convmlp(arch, pretrained, progress, classifier_head, blocks, dims, mlp_ratios, *args, **kwargs):
    model = ConvMLP(blocks=blocks, dims=dims, mlp_ratios=mlp_ratios,
                    classifier_head=classifier_head, *args, **kwargs)
    if pretrained and arch in model_urls:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def convmlp_s(pretrained=False, progress=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_s', pretrained=pretrained, progress=progress,
                    blocks=[2, 4, 2], mlp_ratios=[2, 2, 2], dims=[128, 256, 512],
                    channels=64, n_conv_blocks=2, classifier_head=classifier_head,
                    *args, **kwargs)


def convmlp_m(pretrained=False, progress=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_m', pretrained=pretrained, progress=progress,
                    blocks=[3, 6, 3], mlp_ratios=[3, 3, 3], dims=[128, 256, 512],
                    channels=64, n_conv_blocks=3, classifier_head=classifier_head,
                    *args, **kwargs)


def convmlp_l(pretrained=False, progress=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_l', pretrained=pretrained, progress=progress,
                    blocks=[4, 8, 3], mlp_ratios=[3, 3, 3], dims=[192, 384, 768],
                    channels=96, n_conv_blocks=3, classifier_head=classifier_head,
                    *args, **kwargs)



class SegConvMLP(ConvMLP):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 *args, **kwargs):
        super(SegConvMLP, self).__init__(
            blocks=blocks,
            dims=dims,
            mlp_ratios=mlp_ratios,
            channels=channels,
            n_conv_blocks=n_conv_blocks,
            classifier_head=False,
            *args, **kwargs)


    def forward(self, x):
        outs = []
        x = self.tokenizer(x)
        outs.append(x)
        x = self.conv_stages(x)
        outs.append(x)
        x = x.permute(0, 2, 3, 1)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i != 1 and len(self.stages) == 3:
                outs.append(x.permute(0, 3, 1, 2).contiguous())

        return tuple(outs)

#    def init_weights(self, pretrained):
#        if isinstance(pretrained, str):
#            logger = get_root_logger()
#            load_checkpoint(self, pretrained, strict=False, logger=logger)
#        else:
#            raise TypeError(f'Invalid backbone pretrained URL/Filename: {pretrained}')


class SegConvMLPSmall(SegConvMLP):
    def __init__(self, *args, **kwargs):
        super(SegConvMLPSmall, self).__init__(
            blocks=[2, 4, 2],
            dims=[128, 256, 512],
            mlp_ratios=[2, 2, 2],
            channels=64,
            n_conv_blocks=2)
        self.out_channels = [64, 128, 256, 512]


class SegConvMLPMedium(SegConvMLP):
    def __init__(self, *args, **kwargs):
        super(SegConvMLPMedium, self).__init__(
            blocks=[3, 6, 3],
            dims=[128, 256, 512],
            mlp_ratios=[3, 3, 3],
            channels=64,
            n_conv_blocks=3)
        self.out_channels = [64, 128, 256, 512]


class SegConvMLPLarge(SegConvMLP):
    def __init__(self, *args, **kwargs):
        super(SegConvMLPLarge, self).__init__(
            blocks=[4, 8, 3],
            dims=[192, 384, 768],
            mlp_ratios=[3, 3, 3],
            channels=96,
            n_conv_blocks=3)
        self.out_channels = [96, 192, 384, 768]