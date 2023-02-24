import numpy as np

from augmentations.custom_aug.augment import (
    wave_deform,
    elastic_transform
)

from albumentations.core.transforms_interface import ImageOnlyTransform


class WaveDeform(ImageOnlyTransform):
    """Applies a wave deform augmentation to an image.

    This class is a subclass of ImageOnlyTransform and applies a wave deform
    augmentation to an image. The wave deform is applied using the wave_deform
    function, which distorts the image in a wavy pattern.

    Args and attributes:
        grid_size: The size of the grid used to apply the wave deform.
        sin_base: The angle of the wave deform, chosen randomly from this range.
        alpha: The weight of the wave deform, controlling the intensity of the distortion.

    Typical usage example:

        image = np.load('image.npy')

        # Initialize the WaveDeform object with desired parameters
        wave_deform = WaveDeform(grid_size=[32, 64], sin_base=[5, 10], alpha=0.5)

        # Apply the wave deform augmentation to the image
        augmented_image = wave_deform.apply(image)
    """

    def __init__(self, **kwargs):
        self.grid_size = kwargs.pop("grid_size")
        self.sin_base = kwargs.pop("sin_base")
        self.alpha = kwargs.pop("alpha")
        super().__init__(**kwargs)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Applies the wave deform augmentation to an image.

        Args:
            img: The source image to be augmented.

        Returns:
            np.ndarray: The augmented image.
        """
        return wave_deform(
            src=img,
            grid_size=self.grid_size,
            alpha=self.alpha,
            sin_base=self.sin_base
        )

    def get_transform_init_args_names(self):
        """
        This method is required for albumentations compatability
        """
        return {
            "grid_size": self.grid_size,
            "sin_base": self.sin_base,
            "alpha": self.alpha
        }


class ElasticTransform(ImageOnlyTransform):
    """Applies an elastic transform augmentation to an image.

    This class is a subclass of ImageOnlyTransform and applies an elastic transform
    augmentation to an image. The elastic transform is applied using the elastic_transform
    function, which distorts the image in an elastic-like manner.

    Args and attributes:
        alpha: The weight of the Gaussian filter used in the elastic transform.
        sigma: The parameter of the Gaussian filter used in the elastic transform.
        alpha_affine: The weight of the affine distortion applied in the elastic transform.

    Typical usage example:

        image = np.load('image.npy')

        # Initialize the WaveDeform object with desired parameters
        elastic_transform = ElasticTransform(alpha=0.5, sigma=0.035, alpha_affine=0.015)

        # Apply the wave deform augmentation to the image
        augmented_image = elastic_transform.apply(image)
    """

    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha")
        self.sigma = kwargs.pop("sigma")
        self.alpha_affine = kwargs.pop("alpha_affine")
        super().__init__(**kwargs)

    def apply(self,
              img: np.ndarray,
              **params
              ) -> np.ndarray:
        """
        Applies the elastic transform augmentation to an image.

        Args:
            img: The source image to be augmented.

        Returns:
            np.ndarray: The augmented image.
        """
        height, width, _ = img.shape
        return elastic_transform(
            src=img,
            alpha=self.alpha * width,
            sigma=self.sigma * width,
            alpha_affine=self.alpha_affine * height
        )

    def get_transform_init_args_names(self):
        """
        This method is required for albumentations compatability
        """
        return {
            "alpha": self.alpha,
            "sigma": self.sigma,
            "alpha_affine": self.alpha_affine
        }
