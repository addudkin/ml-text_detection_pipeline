from src.models.heads.db_head import DBHead, DBHeadV2
from src.models.heads.simple_head import ConvHead
from src.models.heads.db_head_without_conv_trans import DBHeadSimpleUpsample

__all__ = ['build_head']
support_head = ['ConvHead', 'DBHead', 'DBHeadV2', 'MaskTextSpotterHead', 'DBHeadSimpleUpsample']


def build_head(head_name, **kwargs):
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head
