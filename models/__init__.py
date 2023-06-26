import torch

from omegaconf import DictConfig
from utils.tools import load_checkpoints
from src.models.detectors.db import DB, DBCism, DBCismEvenOddLines
from torch.nn.parallel.distributed import DistributedDataParallel
import segmentation_models_pytorch as sm


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
    if cfg["model"]["class_model"] in ['Unet', 'DeepLabV3Plus']:
        model = eval(f"sm.{cfg['model']['class_model']}")(**cfg['model']['hparams'])
    else:
        model = eval(cfg["model"]["class_model"])(cfg)

    if cfg["checkpoint"]["path2best"]:
        load_checkpoints(model, cfg['checkpoint']['path2best'])
    # if cfg["model"]["tune_model"] == 'TDLineHead':
    #     main_model = eval(cfg["model"]["class_model"])(cfg)
    #     load_checkpoints(main_model, cfg['checkpoint']['path2best'])
    #     model = TDLineHead(main_model, 256)

    #
    # else:
    #     model = eval(cfg["model"]["class_model"])(cfg)
    #
    # if cfg["model"]["tune_model"] != 'TDLineHead':
    #     if cfg["checkpoint"]["path2best"]:
    #         load_checkpoints(model, cfg['checkpoint']['path2best'])

    model = model.to(device)

    if cfg["train"]["use_ddp"]:
        model = DistributedDataParallel(model,
                                        device_ids=[cfg["rank"]],
                                        output_device=cfg["rank"],
                                        find_unused_parameters=True)

    return model
