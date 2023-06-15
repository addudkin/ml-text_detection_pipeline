import torch

from criterions.db_loss import DBLoss, DBLossV2, DBLossIntersection
from criterions.fieldnet_loss import MultiLoss


def get_criterion(loss_function: dict) -> torch.nn.Module:
    """Creates criterion function.

    Args:
        loss_function: The loss function name and parameters.

    Returns:
        torch.nn.Module, the criterion function.
    """
    if 'DB' in loss_function['type']:
        criterion = eval(loss_function['type'])(**loss_function['params'])
    else:
        criterion = eval(loss_function['type'])(loss_function['params'])

    return criterion
