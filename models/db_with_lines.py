import torch.nn as nn
import torch.nn.functional as F


class TDLineHead(nn.Module):
    def __init__(self, model, in_channels):

        super().__init__()

        self.backbone = model.backbone
        self.neck = model.neck

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.neck.parameters():
            param.requires_grad = False

        inner_channels = in_channels
        bias = False
        smooth = False

        self.line_head = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 2, smooth=smooth, bias=bias)
        )

        self.line_head.apply(self.weights_init)

    def freeze_feature_extractor(self):
        self.backbone.eval()
        self.neck.eval()

        for child in self.backbone.children():
            for param in child.parameters():
                param.requires_grad = False

        for child in self.neck.children():
            for param in child.parameters():
                param.requires_grad = False

    def freeze_feature_backbone(self):
        self.backbone.eval()

        for child in self.backbone.children():
            for param in child.parameters():
                param.requires_grad = False

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, x):
        _, _, H, W = x.size()
        out = self.backbone(x)
        out = self.neck(out)
        out = self.line_head(out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return out