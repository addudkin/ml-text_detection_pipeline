import math

from typing import List, Tuple
from random import randint

import numpy as np


class WaveDeformer:
    """
    Applies a wave transformation to an image.

    This class applies a wave transformation to an image, distorting it in a wavy pattern.
    The wave transformation is defined by a grid size, an alpha parameter, and a range of
    sine angles.

    Args:
        grid_size: The size of the grid cells used to apply the wave transformation,
            specified as a tuple of (width, height). Default is (20, 10).
        alpha: The force parameter multiplied by the sine of the angle. Default is 5.
        sin_base: The range of sine angles to choose from, specified as a tuple of
            (min, max). Default is (10, 45).

    Attributes:
        grid_w (int): Width if the grid cells
        grid_h (int): Height if the grid cells
        sin_base (int): Angel which will apply to sin function

    Example:
        deformer = WaveDeformer()
        img = Image.open('input.jpg')
        img = ImageOps.deform(img, deformer)
        img.save('output.jpg')
    """
    def __init__(self,
            grid_size: Tuple[int, int] = (20, 10),
            alpha: int = 5,
            sin_base: Tuple[int, int] = (10, 45)
    ):
        self.grid_w, self.grid_h = grid_size
        self.alpha = alpha
        self.sin_base = randint(*sin_base)

    def transform(self,
                  x: int,
                  y: int
    ) -> Tuple[int, float]:
        """
        Transforms the specified y coordinates using the wave transformation.

        This function applies the wave transformation to the specified y coordinates. The
        transformation is defined by the alpha parameter, the sine base, and the grid size of
        the `WaveDeformer` instance.

        Args:
            x: The x coordinate to transform.
            y: The y coordinate to transform.

        Returns:
            Tuple[int, float]: The transformed y and x coordinates.
        """
        y = y + self.alpha * math.sin(x / self.sin_base)
        return x, y

    def transform_rectangle(
            self,
            x0: int,
            y0: int,
            x1: int,
            y1: int
    ) -> Tuple[int, float, int, float, int, float, int, float]:
        """
        Transforms the corners of the specified rectangle using the wave transformation.

        This function applies the wave transformation to the corners of the specified rectangle.
        The transformation is defined by the alpha parameter, the sine base, and the grid size
        of the `WaveDeformer` instance.

        Args:
            x0: The x coordinate lower left cell point.
            y0: The y coordinate lower left cell point.
            x1: The x coordinate upper right cell point.
            y1: The y coordinate upper right cell point.

        Returns:
            Tuple[int, float, int, float, int, float, int, float]: The transformed coordinates of the
                four corners of the rectangle, in the order (x0, y0, x0, y1, x1, y1, x1, y0).
        """
        x0y0 = self.transform(x0, y0)
        x0y1 = self.transform(x0, y1)
        x1y1 = self.transform(x1, y1)
        x1y0 = self.transform(x1, y0)

        out = (
            x0y0[0], x0y0[1],
            x0y1[0], x0y1[1],
            x1y1[0], x1y1[1],
            x1y0[0], x1y0[1]
        )
        return out

    def getmesh(
            self, img: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int, int, int, int, int],
                    Tuple[int, float, int, float, int, float, int, float]]]:
        """Generates a mesh of transformed coordinates for the given image.

        This function generates a mesh of transformed coordinates for the given image using the
        wave transformation defined by the alpha parameter, the sine base, and the grid size of
        the `WaveDeformer` instance. The mesh is a list of tuples, where each tuple contains a
        pair of (target, source) coordinates. The target coordinates represent the positions in
        the original image, and the source coordinates represent the positions in the transformed
        image. The mesh can be used to apply the wave transformation to the image using the
        `Image.transform()` method.

        Args:
            img: The image to generate the mesh for.

        Returns:
            List: A list of tuple containing
                pairs of (target, source) coordinates for the mesh.

        """
        img_w, img_h = img.size

        target_grid = []
        for x in range(0, img_w, self.grid_w):
            for y in range(0, img_h, self.grid_h):
                target_grid.append((x, y, x + self.grid_w, y + self.grid_h))

        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

        return [t for t in zip(target_grid, source_grid)]
