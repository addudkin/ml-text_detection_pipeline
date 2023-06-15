import copy

from utils.tools import get_config

import albumentations as A
import augmentations.custom_aug as custom_aug

from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import NoOp

from omegaconf import ListConfig, DictConfig
from typing import Optional, Any, Dict, Union


def get_aug_from_config(
    cfg: DictConfig,
    limit_augs: Optional[DictConfig] = None
) -> Any:
    cfg = copy.deepcopy(cfg)

    """Creates a list of Albumentation augmentations based on a provided configuration.
    
    This function recursively parses a provided configuration dictionary and creates a list of Albumentation
    augmentations based on the provided parameters. The supported modes are: Compose, OneOf, and SomeOf.
    By default, a list of augmentations will be packed in a Compose function.
    
    Args:
        cfg: The configuration dictionary specifying the augmentations to be created.
        limit_augs: A dictionary specifying the maximum number of augmentations that
            can be applied. If not provided, the number of augmentations will not be limited.
    
    Returns:
        Any: A list of Albumentation augmentations based on the provided configuration.
    """

    if cfg is None:
        return NoOp()

    if isinstance(cfg, str):
        return name2factory(cfg)()

    if isinstance(cfg, list) or isinstance(cfg, ListConfig):
        augs = [get_aug_from_config(c, limit_augs) for c in cfg]
        return A.Compose(augs)

    # Augmentation name
    name = list(cfg.keys())[0]
    cfg = cfg[name] if cfg[name] else {}

    # Args for Compose/OneOf
    args = cfg.args
    args = args if args is not None else []

    if name == "Compose":
        return A.Compose([get_aug_from_config(c, limit_augs) for c in args], p=cfg.p)
    elif name == "OneOf":
        return A.OneOf([get_aug_from_config(c, limit_augs) for c in args], p=cfg.p)
    elif name == "SomeOf":
        return A.SomeOf([get_aug_from_config(c, limit_augs) for c in args],p=cfg.p, n=cfg.n)
    else:
        return name2factory(name)(*args, **list2tuple(**cfg))


def name2factory(name: str) -> A.BasicTransform:
    """Gets augmentation class by name.

    Args:
        name: The name of the available augmentation.

    Returns:
        A.BasicTransform, the augmentation class.

    Raises:
        AttributeError: if the augmentation class is not found.
    """
    try:
        print(name)
        if name in ['WaveDeform', 'ElasticTransform_',
                    'DirtyDrumTransform', 'LightingGradientTransform',
                    'BadPhotoCopyTransform']:
            return getattr(custom_aug, name)
        # Get from albumentations.core and albumentations.augmentation
        return getattr(A, name)
    except AttributeError:
        print(f"{name} augmentation will be imported from albumentations.pytorch")
        # Get from albumentations.pytorch
        return getattr(A.pytorch, name)


def list2tuple(**kwargs) -> Dict[str, Union[tuple, Any]]:
    """Converts lists to tuples for proper class initialization.

    Args:
        **kwargs: The keyword arguments to be converted.

    Returns:
        dict, the converted keyword arguments.
    """
    return {
        k: tuple(v) if (isinstance(v, list) or isinstance(v, ListConfig)) and len(v) == 2 else v
        for k, v in kwargs.items()
    }


if __name__ == '__main__':
    path2myproject = '/Users/anton/PycharmProjects/ml-text_recognition_pipeline'
    configs = get_config(default=f'{path2myproject}/configs/pretrain_baseline_special_augs.yml')

    train_transform_config = configs['train_transforms']
    val_transform_config = configs['val_transforms']
    post_transform_config = configs['post_transforms']

    train_transform = get_aug_from_config(train_transform_config)
    val_transform = get_aug_from_config(val_transform_config)
    post_transform = get_aug_from_config(post_transform_config)
    print(train_transform)