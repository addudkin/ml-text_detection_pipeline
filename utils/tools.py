import os
import random
import argparse
from textwrap import dedent

import cv2
import yaml
import json

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd

from omegaconf import OmegaConf, DictConfig
from typing import Tuple, Any, Union
from clearml import Task
from clearml.task import TaskInstance


def set_clearml_task(config: DictConfig):
    # Инициализируем Clearml таску
    task = Task.init(
        project_name=config['description']['project_name'],
        task_name=config['description']['experiment_name'],
    )

    # Устанавливаем конфиг
    task.connect_configuration(config['config_path'])

    # Устанавливаем параметры инициализации
    task.set_parameters_as_dict({
        'path2weight': config["checkpoint"]["path2best"],
        'dataset_id': config['description']['dataset_id']
    })

    # Добавляем shell скрипт в базовые настройки контейнера
    task.set_base_docker(
        docker_setup_bash_script=dedent(
            """\
        #!/bin/bash

        apt update
        apt-get install -y libgl1
        """
        ),
    )

    return task


def get_config(
        default: str = "./configs/base_config.yml"
) -> Tuple[DictConfig, TaskInstance]:
    """Parses command-line arguments and reads a configuration file.

    Args:
        default: The default path to the configuration file. Defaults to "./configs/base_config.yml".

    Returns:
        A dictionary-like object containing the configuration parameters.

    Raises:
        ValueError: If the `LOCAL_RANK` environment variable is not set and the `use_ddp` flag is set to True.
    """
    parser = argparse.ArgumentParser(description='Read config')
    parser.add_argument("-c", "--config", type=str, default=default, help="path to config (YAML file)")
    parser.add_argument("-w", "--checkpoint", required=False, type=str, default=None, help="path to weight for convert to jit")
    parser.add_argument("-a", "--do_average", required=False, type=bool, default=False, help="average best chkp")
    parser.add_argument("-r", "--register", required=False, type=bool, default=False, help="register model during convert")
    parser.add_argument("-d", "--dataset_hash", required=False, type=str, default=False, help="dataset hash")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.full_load(f)

    config['config_path'] = args.config

    config['train']['use_ddp'] = False

    local_rank = os.getenv('LOCAL_RANK')

    if local_rank is not None:
        config['train']['use_ddp'] = True
        world_size = torch.cuda.device_count()
        config['world_size'] = world_size
        config['rank'] = int(local_rank)

    config["main_process"] = True

    if config["train"]["use_ddp"]:
        backend = eval(f"dist.Backend.{config['train']['ddp_backend']}")
        dist.init_process_group(backend, rank=config["rank"], world_size=config["world_size"])
        config["main_process"] = config['rank'] == 0

    config["checkpoint"]["path2best"] = args.checkpoint
    config["checkpoint"]["do_average"] = args.do_average
    config["register_model"] = args.register
    config["dataset_hash"] = args.dataset_hash

    config = OmegaConf.create(config)

    task = None

    if config['init_cleaml']:
        task = set_clearml_task(config)

    return config, task


def get_device(cfg: DictConfig) -> torch.device:
    """Gets the device to be used for training or inference.

    Args:
        cfg: A dictionary-like object containing the configuration parameters.

    Returns:
        A torch.device object representing the device to be used.
    """
    gpu_index = cfg['train']['gpu_index']

    if cfg['train']['use_ddp']:
        gpu_index = cfg["rank"]

    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() and gpu_index is not None else "cpu")

    try:
        torch.cuda.set_device(device)
    except ValueError:
        print(f"Cuda device {device} not found")

    return device


def load_checkpoints(
        model: torch.nn.Module,
        path2checkpoint: str
) -> None:
    """Loads the weights of a model from a checkpoint file.

    Args:
        model: The model whose weights will be loaded.
        path2checkpoint: The path to the checkpoint file.
    """
    print(f"Loading weight from {path2checkpoint}")

    checkpoint = torch.load(path2checkpoint, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if
                       k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("Model is downloaded")


def clean_device(device: torch.device) -> None:
    """Clears the memory cache of a device.

    Args:
        device: The device whose memory cache will be cleared.
    """
    if device.type != "cpu":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def set_random_seed(
        seed: int = None,
        deterministic: bool = True
) -> None:
    """Sets the random seed for the Python, NumPy, and PyTorch random number generators.

    Args:
        seed: The seed to be used. If None, the seed will not be set. Defaults to None.
        deterministic: Whether to set the deterministic option for the CUDNN backend.
            If True, `torch.backends.cudnn.deterministic` will be set to True and `torch.backends.cudnn.benchmark`
            will be set to False. If False, the deterministic option will not be set. Defaults to True.
    """

    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def save_json(
        file: Any,
        path2save: str
) -> None:
    """Save json to the json file.

    Args:
         file: File witch will be saved
         path2save: Local path where json file will be saved
    """
    with open(path2save, 'w') as f:
        json.dump(file, f)


def load_json(
        path2load: str
) -> dict:
    """Load json file.

    Args:
        path2load: Local .json file which will be loaded

    Return:
        Dict with content
    """
    with open(path2load, 'r') as j:
        file = json.load(j)
    return file


def load_file(
        path2load: str
) -> Union[pd.DataFrame, dict]:
    """Load json or pandas file.

    Args:
        path2load: .csv of .json file which will be loaded

    Return:
        Union[pd.DataFrame, dist] variable with content
    """
    assert path2load.endswith(('.json', '.csv')), f"Don't support file type for loading -> {path2load}"
    if path2load.endswith('.json'):
        return load_json(path2load)
    else:
        return pd.read_csv(path2load)


def make_target_mask(image, text_polys):
    heght, width = image.shape[:2]
    mask = np.zeros((heght, width), dtype=np.float32)
    for i in range(len(text_polys)):
        polygon = text_polys[i]
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)

    return mask