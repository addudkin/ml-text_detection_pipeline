import torch
from omegaconf import DictConfig
from typing import Tuple

list_schedulers = [
    'LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ConstantLR', 'LinearLR',
    'ExponentialLR', 'SequentialLR', 'CosineAnnealingLR', 'ChainedScheduler',
    'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'PolynomialLR', 'ReduceLROnPlateau'
] # disable ReduceLROnPlateau (need metric for step)


def get_optimizer_and_scheduler(
        model: torch.nn.Module,
        cfg: DictConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer] :
    """Instantiates an optimizer and a scheduler based on the provided configuration.

    This function instantiates an optimizer and a scheduler based on the configuration provided in the `cfg` argument.
    The function also checks that the specified scheduleris supported by the function.

    Args:
        model: The model to optimize.
        cfg: A dictionary-like configuration object containing the optimizer and scheduler configurations.
    """
    optimizer_cfg = cfg["optimizer"]
    scheduler_cfg = cfg["scheduler"]

    assert scheduler_cfg['type'] in list_schedulers, f"Scheduler {scheduler_cfg['type']} don't supported"

    optimizer = eval(f"torch.optim.{optimizer_cfg['type']}")(model.parameters(),
        **optimizer_cfg['params']
    )

    scheduler = eval(f"torch.optim.lr_scheduler.{scheduler_cfg['type']}")(
        optimizer=optimizer, **scheduler_cfg["params"]
    )

    return optimizer, scheduler
