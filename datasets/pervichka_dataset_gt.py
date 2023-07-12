import os.path
from abc import ABC

import cv2
import torch
import numpy as np

from typing import Tuple, List
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.tools import load_json
from utils.resize import ImageResizer


def load_image(path: str, gray=False) -> np.ndarray:
    image = cv2.imread(path)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class TextDetPervichkaGT(
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
        self.annotation = load_json(
            os.path.join(f"{self.config['data']['datasets']['pervichka']['annotation_folder']}/{self.config['data']['sample_name'][split]}")
        )

        self.images = list(self.annotation.keys())
        self.targets = []
        for bboxes in self.annotation.values():
            self.targets.append([np.array(bbox).reshape(-1, 2) for bbox in bboxes])

        self.resizer = ImageResizer(
            config['resizer']
        )

    def load_sample(
            self,
            idx: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Loads a single sample from the dataset.

        Args:
            idx: The index of the sample to load.

        Returns:
            tuple, containing the image as a NumPy array, the text as a string, and the metadata as a string.
        """

        # TODO: Подгрузка сохраненных масок

        image_path = os.path.join(
            self.config['data']['datasets']['pervichka']['images_folder'],
            self.images[idx]
        )

        image = load_image(image_path)

        targets = self.targets[idx]

        return image, targets

    def __getitem__(self, idx):
        image, text_polys = self.load_sample(idx)

        image_instance = self.resizer.run(image)

        image_tensor = torch.as_tensor(image_instance.image).permute((2, 0, 1))

        image_instance.image = image

        out = dict(
            image=image_tensor,
            image_instance=image_instance,
            text_polys=text_polys,
        )

        return out

    def __len__(self) -> int:
        """Return the len of list_filenames"""
        return len(self.images)