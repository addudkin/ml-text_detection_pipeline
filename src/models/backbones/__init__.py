#from src.models.backbones.mobilenet_v3 import MobileNetV3
#from src.models.backbones.mobilenet_v3_torch import mobilenet_v3_large, mobilenet_v3_small
#from src.models.backbones.mobilenet_v3_torch_quantization import mobilenet_v3_large_quantization
from src.models.backbones.resnet import *
from src.models.backbones.resnet_quantization import *
from src.models.backbones.efficientnetv2 import EfficientNetV2_S, EfficientNetV2_M, EfficientNetV2_L, EfficientNetV2_XL
from src.models.backbones.convmlp import SegConvMLPSmall, SegConvMLPMedium, SegConvMLPLarge
from src.models.backbones.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
#from src.models.backbones.shufflenet_v2 import *

__all__ = ['build_backbone']

support_backbone = ['resnet18', 'deformable_resnet18', 'deformable_resnet50',
                    'resnet50', 'resnet34', 'resnet101', 'resnet152',
                    'EfficientNetV2_S', 'EfficientNetV2_M', 'EfficientNetV2_L',
                    'SegConvMLPSmall', 'SegConvMLPMedium', 'SegConvMLPLarge',
                    'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
                    'EfficientNetV2_XL', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'MobileNetV3',
                    'truncated2_resnet50', 'truncated2_deformable_resnet50',
                    "mobilenet_v3_large", 'mobilenet_v3_large_quantization',
                    'resnet18_quantization', 'resnet34_quantization', 'resnet50_quantization',
                    'resnet101_quantization', 'resnet152_quantization', ]


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
