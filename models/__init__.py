import torch

from omegaconf import DictConfig
from utils.tools import load_checkpoints
from models.dbpp import DBNetpp
from src.models.detectors.db import DB, DBCism
from torch.nn.parallel.distributed import DistributedDataParallel


def get_model(
        cfg: DictConfig,
        device=torch.device('cpu')
) -> torch.nn.Module:
    """Instantiates a model object and loads it from a checkpoint if specified. Optionally wraps the model in a
    DistributedDataParallel object for distributed training on multiple GPUs.

    Args:
        cfg: A dictionary containing model and training configuration information.
        device: A torch device (defaults to `torch.device('cpu').

    Returns:
        The modified model object.
    """
    model = eval(cfg["model"]["class_model"])(cfg)

    if cfg["checkpoint"]["path2best"]:
        load_checkpoints(model, cfg['checkpoint']['path2best'])

    model = model.to(device)

    if cfg["train"]["use_ddp"]:
        model = DistributedDataParallel(model,
                                        device_ids=[cfg["rank"]],
                                        output_device=cfg["rank"],
                                        find_unused_parameters=True)

    return model
