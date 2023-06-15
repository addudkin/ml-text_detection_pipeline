import os.path
from abc import ABC

import cv2
import torch
import numpy as np

from typing import Tuple, List
from omegaconf import DictConfig
from torch.utils.data import Dataset

from augmentations import get_aug_from_config
from prepare_mask import ImageSaver
from utils.tools import load_json
from utils.resize import resize


class TextDetPervichka(
    ABC,
    Dataset
):
    """A dataset class for Optical Character Recognition (OCR) tasks.

    This dataset class extends the `Dataset` class from the `torch.utils.data` module and is used
    for tasks involving Optical Character Recognition (OCR). It handles the loading and preprocessing of the data.

    Args:
        ABC: Abstract base class from the `abc` module.
        Dataset: Base class from the `torch.utils.data` module.
        cfg: A dictionary-like object containing the configuration for the dataset.
        split: The name of the data split (e.g. 'train', 'val', 'test').

    Attributes:
        cfg (DictConfig): A dictionary-like object containing the configuration for the dataset.
        batch_size (int): The batch size for training.
        split (str): The name of the data split (e.g. 'train', 'val', 'test').
        path2images (str): The path to the folder containing the images for the dataset.
        data (pandas.DataFrame): A DataFrame containing the filenames, texts, and metadata for the
            data samples.
        list_filenames (List[str]): A list of filenames for the data samples in the dataset.
        list_texts (List[str]): A list of texts for the data samples in the dataset.
        list_metas (List[str]): A list of metadata strings for the data samples in the dataset.
        meta2label (Dict[str, int]): A dictionary mapping metadata strings to integer labels.
        label2meta (Dict[int, str]): A dictionary mapping integer labels to metadata strings.
        height (int): The height of the images in the dataset.
        width (int): The width of the images in the dataset.
        do_pad_image (bool): A flag indicating whether to pad the images.
        pad_type (str): The padding type to use if `do_pad_image` is `True`.
        """
    def __init__(self,
                 config: DictConfig,
                 split: str):

        super(Dataset, self).__init__()
        self.config = config
        self.split = split
        self.loader = ImageSaver()
        self.transforms = get_aug_from_config(self.config[f"{split}_transforms"])
        self.pre_transforms = get_aug_from_config(self.config[f"{split}_pre_transforms"])

        self.annotation = load_json(
                    os.path.join(f"{self.config['data_preprocess']['data_folder']}/annotation_{split}.json")
                )

    def load_sample(
            self,
            idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a single sample from the dataset.

        Args:
            idx: The index of the sample to load.

        Returns:
            tuple, containing the image as a NumPy array, the text as a string, and the metadata as a string.
        """

        # TODO: Подгрузка сохраненных масок

        image_path = os.path.join(
            self.config['data_preprocess']['data_folder'],
            self.split,
            self.annotation['images'][idx]
        )
        targets_path = os.path.join(
            self.config['data_preprocess']['data_folder'],
            self.split,
            self.annotation['targets'][idx]
        )

        image = self.loader.fast_numpy_load(image_path).astype(np.float32)
        targets = self.loader.fast_numpy_load(targets_path).astype(np.float32)

        return image, targets

    def __getitem__(self, idx):
        image, targets = self.load_sample(idx)

        targets = targets.transpose(1, 2, 0)

        image, targets = resize(
            image=image.astype(np.uint8),
            mask=targets.astype(np.uint8),
            img_size=self.config[f'{self.split}']['image_size'],
            split=self.split.upper()
        )

        image = self.pre_transforms(image=image)['image']
        image = image.astype(np.float32)
        targets = targets.astype(np.float32)

        shrink_maps, shrink_masks, threshold_maps, threshold_masks = targets.transpose(2, 0, 1)
        threshold_maps /= 255

        transformed = self.transforms(
            image=image,
            masks=[shrink_maps, shrink_masks, threshold_maps, threshold_masks]
        )

        image = transformed['image']
        shrink_maps, shrink_masks, threshold_maps, threshold_masks = transformed['masks']

        shrink_masks = np.full_like(shrink_masks, fill_value=1., dtype=shrink_masks.dtype)

        image = torch.as_tensor(image).permute((2, 0, 1))
        shrink_maps = torch.as_tensor(shrink_maps)
        shrink_masks = torch.as_tensor(shrink_masks)
        threshold_maps = torch.as_tensor(threshold_maps)
        threshold_masks = torch.as_tensor(threshold_masks)

        out = dict(
            image=image,
            shrink_maps=shrink_maps,
            shrink_masks=shrink_masks,
            threshold_maps=threshold_maps,
            threshold_masks=threshold_masks,
        )

        return out

    def __len__(self) -> int:
        """Return the len of list_filenames"""
        return len(self.annotation['images'])