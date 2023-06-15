import os.path
import random

import cv2
import torch
import numpy as np
from typing import Tuple, List
from omegaconf import DictConfig

from datasets.base_dataset import TextDetDataset
from utils.tools import load_json
from augmentations.preprocessing import resize_if_needed_even_odd

from augmentations.mmocr.random_crop import TextDetRandomCropV2
from scipy.ndimage import gaussian_filter1d
import albumentations as A
from utils.resize import resize, ImageResizer
from augmentations import get_aug_from_config


def roll_top_left_first(coords: np.array) -> np.array:
    for _ in range(4):
        distances = np.linalg.norm(coords, 2, axis=1)
        is_first = np.argmin(distances) == 0

        if is_first:
            break
        coords = np.roll(coords, 1, 0)
    else:
        raise Exception("Failed to find correct sort")
    return coords


def calculate_euclidean(pnt_1: np.ndarray,
                        pnt_2: np.ndarray) -> np.ndarray:
    """

    :param pnt_1:
    :param pnt_2:

    :return:
    """
    legs = np.power(pnt_1[np.newaxis] - pnt_2, 2)
    distance = np.sqrt(legs[:, 0] + legs[:, 1])

    return distance


def order_four_points(pts: np.ndarray,
                      sort_using_euclidean: bool = True) -> np.ndarray:
    """

    :param pts:
    :param sort_using_euclidean:
    :return:
    """
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    if sort_using_euclidean:
        d = calculate_euclidean(tl, right_most)
    else:
        d = right_most[:, 1]
    tr, br = right_most[np.argsort(d), :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype=np.float32)



class LineSeg:
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

        self.config = config
        self.split = split

        self.datasets_name = []
        self.dataset_prob = []
        self.annotations = {}
        self.images_pathes = []
        self.images_polygons = []
        self.images_labels = []
        self.transforms = get_aug_from_config(self.config[f"{split}_transforms"])
        self.pre_transforms = get_aug_from_config(self.config[f"{split}_pre_transforms"])

        self.cropper = TextDetRandomCropV2(tuple(self.config['train']['crop_size']))
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

    def get_dots(self, point, top):
        rect = cv2.minAreaRect(point)
        (x, y), (h, w), angle = rect
    #     if angle >= 45:
    #         h, w = w, h
        box = cv2.boxPoints(((x, y), (h, w), angle))
        #box = sorted(box , key=lambda k: [k[1], k[0]])
        box = order_four_points(box)
        #box = roll_top_left_first(box)
        if top:
            return box[0], box[1]
        else:
            return box[-1], box[-2]

    def get_straight_line(self, line):

        x_last = line[-1][0]
        y_center = (line[:, 1].max() + line[:, 1].min()) // 2
        start_dot = [0, y_center]
        end_dot = [x_last, y_center]

        new_line = np.array([start_dot, end_dot])
        return new_line

    def get_straight_rest(self, rest, width, mode):

        if mode == 'max':
            y = rest[:, 1].max()
        elif mode == 'min':
            y = rest[:, 1].min()
        else:
            print('error')

        x0 = rest[0][0]

        start_dot = [x0, y]
        end_dot = [width, y]
        new_line = np.array([start_dot, end_dot])
        return new_line

    def compare_two_rows(self, zero_row, first_row, width, highest):

        zero_row = zero_row[zero_row[:, 0].argsort()]
        first_row = first_row[first_row[:, 0].argsort()]

        if zero_row[-1][0] > first_row[-1][0]:
            if highest == 'even':
                mode = 'max'
            else:
                mode = 'min'
            last_x = first_row[-1][0]
            last_y = first_row[-1][1]
            rest = zero_row[zero_row[:, 0] > last_x]
            zero_row = zero_row[zero_row[:, 0] <= last_x]
        else:
            if highest == 'odd':
                mode = 'max'
            else:
                mode = 'min'
            last_x = zero_row[-1][0]
            last_y = zero_row[-1][1]
            rest = first_row[first_row[:, 0] > last_x]
            first_row = first_row[first_row[:, 0] <= last_x]

        new_line = np.vstack([zero_row, first_row])
        new_line = new_line[new_line[:, 0].argsort()]

        if rest.shape[0] > 4:
            rest = self.get_straight_rest(rest, width, mode)
        else:
            rest = np.array([])

        new_line = self.get_straight_line(new_line)

        if rest.shape[0] > 0:
            new_line = np.vstack([new_line, rest])
            first_dot = new_line[0]
            last_dot = new_line[-1]

            new_line[:, 0] = gaussian_filter1d(new_line[:, 0], 1.0)
            new_line[:, 1] = gaussian_filter1d(new_line[:, 1], 1.0)
            new_line = np.vstack([np.array([0, first_dot[1]]),
                                  new_line,
                                  np.array([width, last_dot[1]])]
                                 )
        else:
            first_dot = new_line[0]
            last_dot = new_line[-1]
            new_line = np.vstack([np.array([0, first_dot[1]]),
                                  new_line,
                                  np.array([width, last_dot[1]])]
                                 )
        return new_line

    def get_row_line_first(self, line_even, line_odd, width, sigma=1.5):

        even_row = np.vstack([self.get_dots(i, top=False) for i in line_even])
        odd_row = np.vstack([self.get_dots(i, top=True) for i in line_odd])

        ## even сверху
        new_line = self.compare_two_rows(even_row, odd_row, width, 'even')

        return new_line

    def get_row_line_second(self, line_even, line_odd, width, sigma=1.5):
        even_row = np.vstack([self.get_dots(i, top=True) for i in line_even])
        odd_row = np.vstack([self.get_dots(i, top=False) for i in line_odd])

        ## odd сверху
        new_line = self.compare_two_rows(even_row, odd_row, width, 'odd')

        return new_line

    def get_smoth_signle_bottom_line(self, line, width):
        line = np.vstack([self.get_dots(i, top=False) for i in line])

        max_y = line[:, 1].max()

        new_line = np.array([[0, max_y], [width, max_y]])
        return new_line

    def create_mask(self, polys_even, polys_odd, image):
        width, height = image.shape[:2]

        mask1 = np.zeros((width, height))
        if len(polys_even) < len(polys_odd):
            print('Пум пум')

        full_lines = []
        if len(polys_even) != len(polys_odd):
            for even_line, odd_line in zip(polys_even[:-1], polys_odd):
                line = self.get_row_line_first(even_line, odd_line, width)
                full_lines.append(line)

            for even_line, odd_line in zip(polys_even[1:], polys_odd):
                line = self.get_row_line_second(even_line, odd_line, width)
                full_lines.append(line)

            full_lines.append(self.get_smoth_signle_bottom_line(polys_even[-1], width))
        else:
            for even_line, odd_line in zip(polys_even, polys_odd):
                line = self.get_row_line_first(even_line, odd_line, width)
                full_lines.append(line)

            for even_line, odd_line in zip(polys_even[1:], polys_odd[:-1]):
                line = self.get_row_line_second(even_line, odd_line, width)
                full_lines.append(line)

            full_lines.append(self.get_smoth_signle_bottom_line(polys_odd[-1], width))

        for line in full_lines:
            if line.shape[0] == 0:
                continue
            mask1 = cv2.polylines(mask1.astype(np.int32), [line.astype(np.int32)], False, color=(1, 1, 1),
                                  thickness=5)

        mask2 = (mask1 == 0).astype(np.int32)

        return mask1, mask2

    def get_polys(self, polys, labels):
        polys_even = []
        polys_odd = []
        for line, label in zip(polys, labels):
            if label % 2 == 0:
                line = [np.array(word).reshape(-1, 2) for word in line]
                line = [word for word in line if word.shape[0] > 1]
                line = [self.additional_check(word) for word in line]
                polys_even.append(line)
            else:
                line = [np.array(word).reshape(-1, 2) for word in line]
                line = [word for word in line if word.shape[0] > 1]
                line = [self.additional_check(word) for word in line]
                polys_odd.append(line)
        return polys_even, polys_odd

    def __getitem__(self, idx):

        image, polygons, labels = self.load_sample(idx)

        polys_even, polys_odd = self.get_polys(polygons, labels)

        image, polygons = resize_if_needed_even_odd(
            image,
            [polys_even, polys_odd],
            max_size=640)

        polys_even, polys_odd = polygons

        mask1, mask2 = self.create_mask(polys_even, polys_odd, image)

        transformed_image = self.pre_transforms(image=image)['image']

        masks = np.array([mask1, mask2]).transpose(1, 2, 0)

        transformed_image, masks = resize(
            image=transformed_image.astype(np.uint8),
            mask=masks.astype(np.uint8),
            img_size=(640, 480),
            split=self.split.upper()
        )

        # Применяем основные аугменташки
        transformed = self.transforms(
            image=transformed_image,
            masks=[masks[:, :, 0], masks[:, :, 1]]
        )

        transformed_image = transformed['image']
        mask1, mask2 = transformed['masks']

        masks = np.array([mask1, mask2])
        image = torch.as_tensor(transformed_image).permute((2, 0, 1))
        masks = torch.as_tensor(masks)

        out = dict(
            image=image,
            masks=masks,
        )
        return out

    def __len__(self) -> int:
        """Return the len of list_filenames"""
        if self.split == "train":
            return len(self.images_pathes) // 5
        else:
            return len(self.images_pathes)