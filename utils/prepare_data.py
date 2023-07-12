import os
import json
import boto3
import traceback

import pandas as pd
import numpy as np

from utils.tools import get_config, load_file
from concurrent.futures import ProcessPoolExecutor
from omegaconf import DictConfig
from typing import List, Union, Tuple, Any
from tqdm import tqdm


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


def _download_file(file_name: str) -> None:
    """Download a file from a cloud storage bucket.

        Args:
            file_name: The name of the file to download.
    """
    try:
        image_buket.download_file("/".join([*url2images, file_name]), os.path.join(imPATH, file_name))
    except Exception:
        traceback.print_exc()


def multiprocessing_download(
        urls: List[str],
        workers: int = 32
) -> None:
    """Download a list of files from a cloud storage bucket using multiple processes.

    Args:
        urls: A list of file names to download.
        workers: The number of worker processes to use.
    """
    os.makedirs(imPATH, exist_ok=True)
    process_bar = tqdm(total=len(urls), desc="download image")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(_download_file, urls):
            process_bar.update()
    process_bar.close()


def download_file_markup(
        path2file: str
) -> Union[dict, pd.DataFrame]:
    """Download a file from a cloud storage bucket and return its contents.

    Args:
       path2file: The file path to download.

    Returns:
       The contents of the downloaded file, either as a dictionary or a Pandas DataFrame.
    """
    markup_buket, url2markup = _get_bucket(path2file)
    markup_buket.download_file("/".join(url2markup), os.path.join(anPATH, url2markup[-1]))
    return load_file(os.path.join(anPATH, url2markup[-1]))


def download_markup(
        config: DictConfig
) -> dict:
    """Download annotation files from a cloud storage bucket and return their contents as a dictionary.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        A dictionary containing the contents of the downloaded annotation files.
    """
    os.makedirs(anPATH, exist_ok=True)
    markup = dict()
    if config.markup.splited:
        markup = _download_splited_murkup(config)
    return markup


def prepare_data(
        config: DictConfig,
        markup: dict
) -> Union[Tuple[dict, List], None]:
    """Prepare data for training or evaluation by extracting filenames from the given annotation dictionary.

    Args:
        config: A dictionary containing configuration parameters.
        markup: A dictionary containing annotation data.

    Returns:
        A tuple containing the annotation dictionary and a list of filenames, or None if the annotation data is not splitted.
    """
    # TODO: prepare_data для unnited случая
    if config.markup.splited:
        list_filenames = []
        for split, data in markup.items():
            list_filenames.extend(data['filename'].tolist())
        return markup, list_filenames
    else:
        return None


def _get_bucket(
        path2file: str
) -> Tuple[Any, List]:
    """Get a cloud storage bucket and the file path within it.

   Args:
       path2file: The file path to get the bucket for.

   Returns:
       A tuple containing the bucket and a list of the file path.
   """
    part_path2files = os.path.normpath(path2file).split(os.sep)
    resource = boto3.resource('s3')
    bucket_name = part_path2files[1]
    bucket = resource.Bucket(bucket_name)
    url2file = part_path2files[2:]

    return bucket, url2file


def _download_splited_murkup(
        config: DictConfig
) -> dict:
    """Download annotation files from a cloud storage bucket and return their contents as a dictionary.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        A dictionary containing the contents of the downloaded annotation files.
    """
    markup = {}
    for split, file in config.markup.splited.items():
        if file:  # If not null -> continue
            path2file = os.path.join(config.markup.common_url, file)
            loaded_file = download_file_markup(path2file)
            markup[split] = loaded_file
    return markup


def run(config: DictConfig) -> None:
    """Download and prepare data for training or evaluation.

    Args:
        config: A dictionary containing configuration parameters.
    """
    markup = download_markup(config)

    annotation, filenames = prepare_data(config, markup)

    new_files = list(set(filenames).difference(set(os.listdir(imPATH))))

    multiprocessing_download(new_files)

    if config.markup.union:
        pass  # TODO: Загрузка и подготовка данных без предварительной разбивки


if __name__ == "__main__":
    config = get_config()

    # Set global params
    prepare_data_config = config.prepare_data
    rPATH = config.data.root_dir
    imPATH = config.data.images_folder
    anPATH = config.data.annotation_folder

    assert prepare_data_config.markup.common_url, "common_url wasn't set"
    assert prepare_data_config.image_url, "image_url wasn't set"
    image_buket, url2images = _get_bucket(prepare_data_config.image_url)

    # Or splited or union mode
    assert bool(prepare_data_config.markup.splited) != bool(prepare_data_config.markup.union),\
        "Set only one 'splited' or 'union'"

    assert sum([
        prepare_data_config.distribution.train,
        prepare_data_config.distribution.val,
        prepare_data_config.distribution.test
    ]) == 1, "Not correct distribution"

    run(prepare_data_config)