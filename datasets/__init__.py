from omegaconf import DictConfig

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from datasets.base_dataset import TextDetDataset
from datasets.multi_dataset import MultiTDDataset
from datasets.cism_two_lines_dataset import MultiTDDatasetTwoLines
from datasets.line_segmentation import LineSeg
from datasets.clasic_db_lines import LinesDataset
from datasets.line_segmentation_only_text import LineSegOnlyText
from datasets.cism_two_lines_and_lines_dataset import EvenOddLine
from datasets.text_and_lines_dataset import TextDetLines
from datasets.even_odd_dataset import TextDetEvenOdd
from datasets.pervichka_dataset import TextDetPervichka
from datasets.collate_fnc import collate_synt, collate_synt_line_det, collate_even_odd, collate_text_and_lines


def get_dataset(
        split: str,
        cfg: DictConfig
) -> TextDetDataset:
    """Initializes dataset class.

    Args:
        split: The sample split.
        cfg: The config with required parameters.

    Returns:
        OCRDataset, the initialized dataset class.
    """
    return eval(cfg['dataset_name'])(cfg, split)


def get_dataloader(
        dataset: TextDetDataset,
        cfg: DictConfig
) -> DataLoader:
    """Initializes dataloader class.

    Args:
        dataset: The torch dataset class.
        cfg: The config with required parameters.

    Returns:
        DataLoader, the initialized dataloader class.
    """
    shuffle = dataset.split == 'train'
    sampler = None

    if cfg['train']['use_ddp'] and dataset.split == "train":
        # TODO: Что за world_size?
        sampler = DistributedSampler(dataset,
                                     num_replicas=cfg['world_size'],
                                     rank=cfg['rank'],
                                     shuffle=shuffle,
                                     seed=42)

    loader = DataLoader(dataset,
                        batch_size=cfg[dataset.split]["batch_size"],
                        shuffle=not cfg['train']['use_ddp'],
                        sampler=sampler,
                        num_workers=cfg["train"]["workers"],
                        pin_memory=cfg["train"]["pin_memory"],
                        collate_fn=collate_synt)

    return loader