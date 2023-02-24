from src.models.necks.fpn.fpn import FPN, Truncated2FPN
from src.models.necks.fpn.fpn2 import FPN2, Truncated2FPN
from src.models.necks.fpn.fpn_bench import FPNDconv
from src.models.necks.fpn.fpn_carafe import FPNCarafe
from src.models.necks.fpn.fpn_lcau import FPNLcau
from src.models.necks.fpn.fpn_segformer import SegFormerFPN

__all__ = ['build_neck']
support_neck = ['FPN', 'Truncated2FPN', 'FPN2', 'FPNDconv', 'FPNCarafe', 'FPNLcau', 'SegFormerFPN']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
