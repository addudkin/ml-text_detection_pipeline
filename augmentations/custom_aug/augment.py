"""
This code is refer from:
https://github.com/RubanSeven/Text-Image-Augmentation-python/blob/master/augment.py

"""

import cv2
import numpy as np

from typing import List, Tuple
from .wavetr import WaveDeformer
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def wave_deform(
        src: np.ndarray,
        grid_size: Tuple[int, int],
        alpha: int,
        sin_base: Tuple[int, int],
) -> np.ndarray:
    """
    Applies a wave deform augmentation to an image.

    This function applies a wave deform augmentation to an image using the
    WaveDeformer class and the ImageOps.deform function. The wave deform
    distorts the image in a wavy pattern.

    Args:
        src: The source image to be augmented.
        grid_size: The size of the grid used to apply the wave deform.
        alpha: The weight of the wave deform, controlling the intensity of the distortion.
        sin_base: The angle of the wave deform, chosen randomly from this range.

    Returns:
        np.ndarray: The augmented image, with the wave deform applied.
    """
    trans = WaveDeformer(
        grid_size=grid_size,
        alpha=alpha,
        sin_base=sin_base
    )

    img = Image.fromarray(src)
    res = ImageOps.deform(img, trans)

    return np.asarray(res, dtype=np.uint8)


def elastic_transform(
        src: np.ndarray,
        alpha: float,
        sigma: float,
        alpha_affine: float,
        random_state=None
) -> np.ndarray:
    """
    Applies an elastic transform augmentation to an image.

    This function applies an elastic transform augmentation to an image, as described in
    [Simard2003]_. The elastic transform distorts the image in an elastic-like manner,
    using a combination of random affine transformations and Gaussian filtering.

    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Args:
        src: The source image to be augmented.
        alpha: The weight of the Gaussian filter.
        sigma: The parameter of the Gaussian filter.
        alpha_affine: The weight of the affine distortion applied.
        random_state: The random seed to use for generating random affine transformations.
            If not provided, a random seed will be chosen.

    Returns:
        np.ndarray: The augmented image, with the elastic transform applied.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = src.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    src = cv2.warpAffine(src, M, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(src, indices, order=1, mode='reflect').reshape(shape)
