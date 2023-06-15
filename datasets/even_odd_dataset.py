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


class TextDetEvenOdd(
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
            os.path.join(
                config['data_preprocess']['data_folder'],
                f'annotation_{self.split}.json'
            )
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

        shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even = targets[0]
        shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd = targets[1]

        targets = np.array([shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even,
                            shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd]).transpose(1, 2, 0)

        image, targets = resize(
            image=image.astype(np.uint8),
            mask=targets.astype(np.uint8),
            img_size=self.config['train']['image_size'],
            split=self.split.upper()
        )

        image = self.pre_transforms(image=image)['image']
        image = image.astype(np.float32)
        targets = targets.astype(np.float32)

        targets = targets.transpose(2, 0, 1)

        shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even = targets[:4]
        threshold_maps_even /= 255

        shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd = targets[4:]
        threshold_maps_odd /= 255

        transformed = self.transforms(
            image=image,
            masks=[shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even,
                   shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd]
        )

        image = transformed['image']
        targets = transformed['masks']
        shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even = targets[:4]
        shrink_masks_even = np.full_like(shrink_masks_even, fill_value=1., dtype=shrink_maps_even.dtype)

        shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd = targets[4:]
        shrink_masks_odd = np.full_like(shrink_masks_odd, fill_value=1., dtype=shrink_maps_odd.dtype)

        even_targets = torch.as_tensor(np.array(
            [shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even]
        ))

        odd_targets = torch.as_tensor(np.array(
            [shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd]
        ))

        image = torch.as_tensor(image).permute((2, 0, 1))

        out = dict(
            image=image,
            even_targets=even_targets,
            odd_targets=odd_targets,
        )

        return out

    def __len__(self) -> int:
        """Return the len of list_filenames"""
        return len(self.annotation['images'])