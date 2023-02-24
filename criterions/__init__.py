import torch

from criterions.db_loss import DBLoss, DBLossV2


def get_criterion(loss_function: dict) -> torch.nn.Module:
    """Creates criterion function.

    Args:
        loss_function: The loss function name and parameters.

    Returns:
        torch.nn.Module, the criterion function.
    """
    criterion = eval(loss_function['type'])(**loss_function['params'])

    return criterion
