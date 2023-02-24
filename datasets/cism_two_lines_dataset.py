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
from augmentations.preprocessing import random_crop, pad_if_neededV2

from augmentations.mmocr.random_crop import TextDetRandomCropV2
import albumentations as A


class MultiTDDatasetTwoLines(
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

        super(MultiTDDatasetTwoLines, self).__init__(config, split)
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
        self.images_labels = []
        self.cropper = TextDetRandomCropV2(tuple(self.config['train']['crop_size']))
        self.resie = A.LongestMaxSize(max_size=1024)
        for k, v in self.config['data']['datasets'].items():
            self.datasets_name.append(k)
            self.dataset_prob.append(v['p'])
            if k not in self.annotations:
                self.annotations[k] = {
                    "images_pathes": list(),
                    "images_polygons": list(),
                    "images_labels": list()
                }

            # annotation = load_json(
            #     os.path.join(f"{v['annotation_folder']}/{self.split}_DS_1685_type2.json")
            # )
            annotation = load_json(
                os.path.join(f"{v['annotation_folder']}/{self.split}_2202_annot.json")
            )

            path2images = v['images_folder']
            images_pathes, images_polygon, images_labels = self._prepare_samples(annotation, path2images)
            self.images_pathes.extend(images_pathes)
            self.images_polygons.extend(images_polygon)
            self.images_labels.extend(images_labels)

            if self.split == 'train':
                self.annotations[k]['images_pathes'].extend(images_pathes)
                self.annotations[k]['images_polygons'].extend(images_polygon)
                self.annotations[k]['images_labels'].extend(images_labels)

    def additional_check(self, box):
        """Сюда попадают в основном боксы переноса строки. Формирую 4 точки из 2 и чуть расширяю"""
        if box.shape[0] != 2:
            return box

        x0y0 = box[0]
        x2y2 = box[1]

        x0y0 = [int(x0y0[0] - 5), int(x0y0[1] - 5)]
        x2y2 = [int(x2y2[0] + 5), int(x2y2[1] + 5)]

        x1y1 = [x2y2[0], x0y0[1]]

        x3y3 = [x0y0[0], x2y2[1]]

        return np.array([x0y0, x1y1, x2y2, x3y3])

    def _prepare_samples(self, annotation, path2images):
        images_pathes = []
        images_polygons = []
        images_labels = []
        for img_name, items in annotation.items():
            path2image = os.path.join(path2images, img_name)
            images_pathes.append(path2image)
            polys = []
            labels = []
            for line, label in zip(items['polys'], items['lines_label']):
                labels.append(label)
                polys.append(line)
            images_polygons.append(polys)
            images_labels.append(labels)
        return images_pathes, images_polygons, images_labels

    def corted_coords(self, transformed_poly_group):
        """Сортируем строки после кропа"""
        if len(transformed_poly_group[0]) == 0 and len(transformed_poly_group[1]) != 0:
            return [transformed_poly_group[1], transformed_poly_group[0]]

        if len(transformed_poly_group[0]) != 0 and len(transformed_poly_group[1]) == 0:
            return [transformed_poly_group[0], transformed_poly_group[1]]

        if len(transformed_poly_group[0]) == 0 and len(transformed_poly_group[1]) == 0:
            return transformed_poly_group

        first_even = transformed_poly_group[0][0]
        first_odd = transformed_poly_group[1][0]

        y_even = first_even[:, 1]
        y_even = y_even.min()

        y_odd = first_odd[:, 1]
        y_odd = y_odd.min()

        if y_even < y_odd:
            even_coords = transformed_poly_group[0]
            odd_cords = transformed_poly_group[1]
        else:
            even_coords = transformed_poly_group[1]
            odd_cords = transformed_poly_group[0]
        return [even_coords, odd_cords]

    def load_sample(
            self,
            idx: int
    ) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
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
            labels =  self.annotations[dataset_name]['images_labels'][data_idx]
        else:
            image_path = self.images_pathes[idx]
            polygons = self.images_polygons[idx]
            labels = self.images_labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, polygons, labels

    def __getitem__(self, idx):

        image, polygons, labels = self.load_sample(idx)

        polys_even = []
        polys_odd = []
        for line, label in zip(polygons, labels):
            if label%2 == 0:
                for word in line:
                    p = np.array(word).reshape(-1, 2)
                    if p.shape[0]>1:
                        polys_even.append(p)
            else:
                for word in line:
                    p = np.array(word).reshape(-1, 2)
                    if p.shape[0]>1:
                        polys_odd.append(p)

        polys_even = [self.additional_check(i) for i in polys_even]
        polys_odd = [self.additional_check(i) for i in polys_odd]

        # create crop
        transformed_image, transformed_poly = pad_if_neededV2(
            image,
            [polys_even, polys_odd] # Группа полигонов
        )

        # if self.split == 'train':
        #     transformed_poly_group = []
        #     for gr in transformed_poly:
        #         transformed_poly_group.append([i.flatten() for i in gr])
        #
        #     cropper_input = {
        #         'img': transformed_image,
        #         'gt_polygons': transformed_poly_group,
        #     }
        #     answer = self.cropper.transform(cropper_input)
        #     transformed_image = answer['img']
        #     transformed_poly = answer['gt_polygons']

        transformed_poly_group = []
        for gr in transformed_poly:
            transformed_poly_group.append([np.array(i).reshape(-1, 2).astype(np.int32) for i in gr])

        transformed_poly_group = self.corted_coords(transformed_poly_group)

        # Создаем таргеты для четных
        shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even = self.target_func.create_target(
            transformed_image,
            transformed_poly_group[0]
        )

        # Создаем таргеты для не четных
        shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd = self.target_func.create_target(
            transformed_image,
            transformed_poly_group[1]
        )

        # Применяем основные аугменташки
        transformed = self.transforms(
            image=transformed_image,
            masks=[shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even,
                   shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd]
        )

        transformed_image = transformed['image']
        shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even,\
            shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd = transformed['masks']

        # Ресайзим до 1024 с сохранением соотношения сторон
        transformed = self.resie(
            image=transformed_image,
            masks=[shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even,
                   shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd]
        )

        transformed_image = transformed['image']
        shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even,\
            shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd = transformed['masks']

        shrink_masks_even = np.full_like(shrink_masks_even, fill_value=1., dtype=shrink_masks_even.dtype)
        shrink_masks_odd = np.full_like(shrink_masks_odd, fill_value=1., dtype=shrink_masks_odd.dtype)

        image = torch.as_tensor(transformed_image).permute((2, 0, 1))

        shrink_maps_even = torch.as_tensor(shrink_maps_even)
        shrink_masks_even = torch.as_tensor(shrink_masks_even)
        threshold_maps_even = torch.as_tensor(threshold_maps_even)
        threshold_masks_even = torch.as_tensor(threshold_masks_even)

        shrink_maps_odd = torch.as_tensor(shrink_maps_odd)
        shrink_masks_odd = torch.as_tensor(shrink_masks_odd)
        threshold_maps_odd = torch.as_tensor(threshold_maps_odd)
        threshold_masks_odd = torch.as_tensor(threshold_masks_odd)

        out = dict(
            image=image,
            target_even=[shrink_maps_even, shrink_masks_even, threshold_maps_even, threshold_masks_even],
            target_odd=[shrink_maps_odd, shrink_masks_odd, threshold_maps_odd, threshold_masks_odd],
            gt_polygons=transformed_poly,
        )
        return out

    def __len__(self) -> int:
        """Return the len of list_filenames"""

        return len(self.images_pathes)