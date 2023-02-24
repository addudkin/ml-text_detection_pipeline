import os.path
import random
from abc import ABC

import cv2
import torch
import numpy as np
from shapely import Polygon
from typing import Tuple, List
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datasets.utils.target_creator import TargetCreator
from datasets.base_dataset import TextDetDataset
from utils.tools import load_json
from augmentations.preprocessing import random_crop, pad_if_needed
from augmentations import get_aug_from_config


class MultiTDDataset(
    TextDetDataset
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

        super(MultiTDDataset, self).__init__(config, split)
        self.config = config
        self.split = split

        self.target_func = TargetCreator(
            self.config['mask_shrinker']['params'],
            self.config['border_creator']['params']
        )

        self.datasets_name = []
        self.dataset_prob = []
        self.annotations = {}
        self.images_pathes = []
        self.images_polygons = []

        for k, v in self.config['data']['datasets'].items():
            self.datasets_name.append(k)
            self.dataset_prob.append(v['p'])
            if k not in self.annotations:
                self.annotations[k] = {
                    "images_pathes": list(),
                    "images_polygons": list()
                }

            # annotation = load_json(
            #     os.path.join(f"{v['annotation_folder']}/{self.split}_DS_1685_type2.json")
            # )
            annotation = load_json(
                os.path.join(f"{v['annotation_folder']}/{self.split}_2202_annot.json")
            )
            path2images = v['images_folder']
            images_pathes, image_polygon = self._prepare_samples(annotation, path2images)
            self.images_pathes.extend(images_pathes)
            self.images_polygons.extend(image_polygon)
            if self.split == 'train':
                self.annotations[k]['images_pathes'].extend(images_pathes)
                self.annotations[k]['images_polygons'].extend(image_polygon)

    # def _prepare_samples(self, annotation, path2images):
    #     images_pathes = []
    #     image_polygons = []
    #     for sample in annotation['data_list']:
    #         img_path = sample['img_path']
    #         img_name = os.path.split(img_path)[-1]
    #         path2image = os.path.join(path2images, img_name)
    #         instances = sample["instances"]
    #         polygons = [np.array(i['polygon']).reshape(-1, 2) for i in instances]
    #         images_pathes.append(path2image)
    #         image_polygons.append(polygons)
    #     return images_pathes, image_polygons

    def additional_check(self, box):

        x0y0 = box[0]
        x2y2 = box[1]

        x0y0 = [int(x0y0[0] - 5), int(x0y0[1] - 5)]
        x2y2 = [int(x2y2[0] + 5), int(x2y2[1] + 5)]

        x1y1 = [x2y2[0], x0y0[1]]

        x3y3 = [x0y0[0], x2y2[1]]

        return np.array([x0y0, x1y1, x2y2, x3y3])

    def _prepare_samples(self, annotation, path2images):
        images_pathes = []
        image_polygons = []
        for img_name, items in annotation.items():
            path2image = os.path.join(path2images, img_name)
            polys = []
            for line in items['polys']:
                for word in line:
                    p = np.array(word).reshape(-1, 2)
                    if p.shape[0] == 2:
                        p = self.additional_check(p)
                    polys.append(p)


            images_pathes.append(path2image)
            image_polygons.append(polys)
        return images_pathes, image_polygons

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
        if self.split == 'train':
            dataset_name = random.choices(self.datasets_name, weights=self.dataset_prob, k=1)[0]
            data_idx = np.random.randint(0, self.__len__())
            image_path = self.annotations[dataset_name]['images_pathes'][data_idx]
            polygons = self.annotations[dataset_name]['images_polygons'][data_idx]
        else:
            image_path = self.images_pathes[idx]
            polygons = self.images_polygons[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, polygons, image_path

    def __len__(self) -> int:
        """Return the len of list_filenames"""

        return len(self.images_pathes)