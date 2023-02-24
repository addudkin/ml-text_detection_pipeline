import os.path
from abc import ABC

import cv2
import torch
import numpy as np

from typing import Tuple, List
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datasets.utils.target_creator import TargetCreator
from utils.tools import load_json
from augmentations.preprocessing import random_crop, pad_if_needed
from augmentations import get_aug_from_config
from augmentations.mmocr.random_crop import TextDetRandomCrop


class TextDetDataset(
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

        self.target_func = TargetCreator(
            self.config['mask_shrinker']['params'],
            self.config['border_creator']['params']
        )

        self.cropper = TextDetRandomCrop(tuple(self.config['train']['crop_size']))
        self.transforms = get_aug_from_config(self.config[f"{split}_transforms"])
        # self.path2images = self.config['data']['images_folder']
        # annotation = load_json(
        #     os.path.join(f"{self.config['data']['annotation_folder']}/{self.split}_instance.json")
        # )
        # self.images_pathes = []
        # self.image_polygons = []
        # for sample in annotation['data_list']:
        #     img_path = sample['img_path']
        #     img_name = os.path.split(img_path)[-1]
        #     path2image = os.path.join(self.path2images, img_name)
        #     instances = sample["instances"]
        #     polygons = [np.array(i['polygon']).reshape(-1, 2) for i in instances]
        #     self.images_pathes.append(path2image)
        #     self.image_polygons.append(polygons)

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
        image_path = self.images_pathes[idx]
        polygons = self.image_polygons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, polygons, image_path

    def __getitem__(self, idx):

        image, polygons, image_path = self.load_sample(idx)

        # create crop
        transformed_image, transformed_poly = pad_if_needed(
            image,
            polygons
        )
        if self.split == 'train':
            transformed_poly = [i.flatten() for i in transformed_poly]

            cropper_input = {
                'img': transformed_image,
                'gt_polygons': transformed_poly,
                'gt_bboxes_labels': np.array([0 for i in range(len(transformed_poly))]),
                'gt_ignored': np.array([False for i in range(len(transformed_poly))])
            }
            answer = self.cropper.transform(cropper_input)
            transformed_image = answer['img']
            transformed_poly = answer['gt_polygons']

        transformed_poly = [np.array(i).reshape(-1, 2).astype(np.int32) for i in transformed_poly]

        shrink_maps, shrink_masks, threshold_maps, threshold_masks = self.target_func.create_target(
            transformed_image,
            transformed_poly
        )

        transformed = self.transforms(
            image=transformed_image,
            masks=[shrink_maps, shrink_masks, threshold_maps, threshold_masks]
        )

        transformed_image = transformed['image']
        shrink_maps, shrink_masks, threshold_maps, threshold_masks = transformed['masks']
        shrink_masks = np.full_like(shrink_masks, fill_value=1., dtype=shrink_masks.dtype)

        image = torch.as_tensor(transformed_image).permute((2, 0, 1))
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
            gt_polygons=transformed_poly,
        )

        return out

    def __len__(self) -> int:
        """Return the len of list_filenames"""
        return len(self.path2images)