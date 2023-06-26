import os
import cv2
import json

import numpy as np
from utils.lines_target_utils import create_lines_mask

from datasets.utils.target_creator import TargetCreator
from omegaconf import DictConfig
from typing import Tuple, List, Any, Union
from utils.tools import load_json, save_json
from tqdm import tqdm
from itertools import repeat


from augmentations.preprocessing import resize_if_needed_even_odd
from concurrent.futures import ProcessPoolExecutor


class ImageSaver:
    @staticmethod
    def fast_numpy_save(array: np.ndarray,
                        path2file: str) -> None:
        with open(path2file, 'wb') as file:
            file.write(array.dtype.name.encode() + b"\n")
            file.write(json.dumps(array.shape).encode() + b"\n")
            file.write(array.tobytes())

    @staticmethod
    def fast_numpy_load(path2file: str) -> np.ndarray:
        with open(path2file, "rb") as file:
            dtype = np.dtype(file.readline().strip())
            shape = json.loads(file.readline().strip().decode())
            buffer = file.read()
        return np.ndarray(shape, dtype=dtype, buffer=buffer)


def _additional_check(box: np.ndarray) -> np.ndarray:
    """
    Create box from line from [1:2] array
    """
    x0y0 = box[0]
    x2y2 = box[1]

    x0y0 = [int(x0y0[0] - 5), int(x0y0[1] - 5)]
    x2y2 = [int(x2y2[0] + 5), int(x2y2[1] + 5)]

    x1y1 = [x2y2[0], x0y0[1]]

    x3y3 = [x0y0[0], x2y2[1]]

    return np.array([x0y0, x1y1, x2y2, x3y3])


def get_bboxes(x: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(x)
    box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    #print(box.shape)
    # #rect = cv2.minAreaRect(x)
    # box = cv2.boxPoints(rect)
    return box


class DBAnnotator:
    """
    DBNet mask preprocessing
    """
    def __init__(self, config: DictConfig):

        self.config = config
        self.splits = ['train', 'val']
        self.images_pathes = {split: [] for split in self.splits}
        self.images_polygons_even = {split: [] for split in self.splits}
        self.images_polygons_odd = {split: [] for split in self.splits}
        self.images_names = {split: [] for split in self.splits}

        for k, v in self.config['data']['datasets'].items():
            for split in self.splits:
                annotation = load_json(
                    os.path.join(f"{v['annotation_folder']}/{split}_markup.json")
                )

                path2images = v['images_folder']

                images_pathes, image_polygons_even, image_polygons_odd, images_name = self._prepare_samples(
                    annotation, path2images
                )

                self.images_pathes[split].extend(images_pathes)
                self.images_polygons_even[split].extend(image_polygons_even)
                self.images_polygons_odd[split].extend(image_polygons_odd)
                self.images_names[split].extend(images_name)


    def _prepare_samples(
            self,
            annotation: dict,
            path2images: str) -> Tuple[
        List[str],
        List[List[List[Union[np.ndarray, Any]]]],
        List[List[List[Union[np.ndarray, Any]]]],
        List[Any]
    ]:
        """Create prepared samples"""
        images_pathes = []
        images_polygons_even = []
        images_polygons_odd = []
        images_name = []

        for img_name, items in annotation.items():
            path2image = os.path.join(path2images, img_name)
            image_even_polys = []
            image_odd_polys = []
            for line, label in zip(items['polys'], items['labels']):
                line_polys = []

                # Transform polys from line
                for word in line:
                    p = np.array(word).reshape(-1, 2)
                    if p.shape[0] == 2:
                        p = _additional_check(p)
                    if p.shape[0] <= 2:
                        continue
                    line_polys.append(p)
                # Add to concreate list by label
                if label == 'even':
                    image_even_polys.append(line_polys)
                else:
                    image_odd_polys.append(line_polys)

            images_polygons_even.append(image_even_polys)
            images_polygons_odd.append(image_odd_polys)
            images_pathes.append(path2image)
            images_name.append(img_name)

        return images_pathes, images_polygons_even, images_polygons_odd, images_name


def prepare_dir(config: DictConfig, splits: List[str]) -> dict:
    """Create save dir for each split"""
    save_dirs = {split: str() for split in splits}
    for split in splits:
        path2dir = os.path.join(config['data_preprocess']['data_folder'], split)
        os.makedirs(path2dir, exist_ok=True)
        save_dirs[split] = path2dir

    return save_dirs


def load_image(
        image_path: str
) -> np.ndarray:
    """Load image """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def check_size_polys(list_polys: List[np.ndarray]) -> List[np.ndarray]:
    """Check poly size"""
    list_polys = [poly for poly in list_polys if poly.shape[0] > 2]
    return list_polys


def get_masks(target_func, image: np.ndarray, polygons: List[List[np.ndarray]]) -> np.ndarray:
    """Crete targets"""

    polys_even, polys_odd = polygons
    mask_line1, mask_line2 = create_lines_mask(polys_even, polys_odd, image)

    all_polygons = []
    for line in polys_even:
        all_polygons.extend(line)
    for line in polys_odd:
        all_polygons.extend(line)

    all_polygons = check_size_polys(all_polygons)
    #all_polygons = [get_bboxes(poly) for poly in all_polygons]

    shrink_maps, shrink_masks, threshold_maps, threshold_masks = target_func.create_target(
        image,
        all_polygons
    )

    threshold_maps *= 255

    return np.array([shrink_maps, shrink_masks, threshold_maps, threshold_masks, mask_line1, mask_line2])


def process_single_sample(event: Tuple) -> Tuple[str, str]:
    """Processing one element"""

    image_path, image_polygons_even, image_polygons_odd, image_name, split = event
    # Load image, polygons
    image = load_image(image_path)

    # Resize and check
    image, polygons = resize_if_needed_even_odd(image, [image_polygons_even, image_polygons_odd], max_size=640)

    image_polygons_even, image_polygons_odd = polygons

    # Get targets
    targets = get_masks(target_func, image, [image_polygons_even, image_polygons_odd])

    # Create new name and add into new annotation
    file_name, _ = os.path.splitext(image_name)
    new_image_name = f'{file_name}_image.npy'
    new_target_name = f'{file_name}_targets.npy'

    # Save result
    saver.fast_numpy_save(
        image.astype(np.uint8),
        os.path.join(save_dirs[split], new_image_name)
    )

    saver.fast_numpy_save(
        targets.astype(np.uint8),
        os.path.join(save_dirs[split], new_target_name)
    )

    return new_image_name, new_target_name,


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import yaml

    splits = ['train', 'val']

    workers = 12
    path2config_db = '/home/addudkin/ml-text_detection_pipeline/configs/general_td_new_mean_std_big_crops.yml'

    with open(path2config_db) as f:
        config_db = yaml.full_load(f)
    config_db = OmegaConf.create(config_db)
    save_dirs = prepare_dir(config_db, splits)

    annotator = DBAnnotator(config_db)

    target_func = TargetCreator(
        config_db['mask_shrinker']['params'],
        config_db['border_creator']['params']
    )

    images_pathes_dict = annotator.images_pathes.copy()
    images_polygons_dict_even = annotator.images_polygons_even.copy()
    images_polygons_dict_odd = annotator.images_polygons_odd.copy()
    images_names_dict = annotator.images_names.copy()
    saver = ImageSaver()

    for split in splits:
        annotation = {
            'images': [],
            'targets': []
        }
        images_pathes = images_pathes_dict[split]
        images_polygons_even = images_polygons_dict_even[split]
        images_polygons_odd = images_polygons_dict_odd[split]
        images_names = images_names_dict[split]

        print(f'Ran {split} processing')
        idxes = [i for i in range(len(images_pathes))]

        # Run Multiprocessing
        process_bar = tqdm(total=len(idxes), desc="Processing markup")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for new_image_name, new_target_name in executor.map(
                    process_single_sample,
                    zip(images_pathes, images_polygons_even, images_polygons_odd, images_names, repeat(split))
            ):
                annotation['images'].append(new_image_name)
                annotation['targets'].append(new_target_name)
                process_bar.update()

        save_json(
            annotation,
            os.path.join(config_db['data_preprocess']['data_folder'], f'annotation_{split}.json')
        )