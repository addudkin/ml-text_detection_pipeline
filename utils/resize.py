import cv2
import numpy as np
from typing import Tuple

from utils.types import ImageResizerResult


class ImageResizer:
    def __init__(self,
                 config) -> None:

        self.input_size = config.get("input_size", (640, 640))  # height, width
        self.aspect_ration = config.get("aspect_ratio", True)
        self.padding_type = config.get("padding_type", "right").upper()
        self.padding_value = config.get("padding_value", 0)
        self.resize_to_height = config.get("resize_to_height", False)
        self.resize_to_power = config.get("resize_to_power", False)
        self.interpolation_method = getattr(cv2, config.get("interpolation_method", "inter_nearest").upper())

        assert self.padding_type in ("RIGHT", "LEFT", "BOTH", ""), "Incorrect padding type"

    def get_new_size(self, image: np.ndarray) -> Tuple[int, int]:

        img_h, img_w = image.shape[:2]
        new_h, new_w = self.input_size

        if self.resize_to_height:
            resized_width, resized_height = int(max(1, img_w * new_h / img_h)), new_h

        else:
            if img_h > img_w:
                scale = new_h / img_h
                resized_height = new_h
                resized_width = int(img_w * scale)
            else:
                scale = new_w / img_w
                resized_height = int(img_h * scale)
                resized_width = new_w

        return min(new_w, resized_width), min(new_h, resized_height)

    def get_new_coords(self,
                       resized_width: float,
                       resized_height: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        """
        new_h, new_w = self.input_size
        if self.padding_type == "RIGHT":
            x1, y1 = 0, 0
            x2, y2 = resized_width, resized_height
        elif self.padding_type == "LEFT":
            x1, y1 = new_w - resized_width, new_h - resized_height
            x2, y2 = new_w, new_h
        elif self.padding_type == "BOTH":
            x1 = (new_w - resized_width) // 2
            y1 = (new_h - resized_height) // 2
            x2 = new_w - x1
            y2 = new_h - y1
            x2 -= (x2 - x1 - resized_width)
            y2 -= (y2 - y1 - resized_height)
        else:
            x1, y1 = 0, 0
            x2, y2 = new_w, new_h

        return (y1, y2), (x1, x2)

    def resize_with_padding(self,
                            image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        """
        resized_height, resized_width = image.shape[:2]
        (y1, y2), (x1, x2) = self.get_new_coords(resized_width, resized_height)

        new_image = np.pad(image,
                           [(y1, self.input_size[0] - y2),
                            (x1, self.input_size[1] - x2),
                            (0, 0)],
                           mode='constant',
                           constant_values=self.padding_value)

        return new_image, (resized_width, resized_height)

    def resize_with_aspect(self,
                           image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        """
        resized_width, resized_height = self.get_new_size(image)

        image = cv2.resize(image, (resized_width, resized_height), self.interpolation_method)

        return image, (resized_width, resized_height)

    def resize_no_aspect(self,
                         image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        """
        new_size = (self.input_size[1], self.input_size[0])
        image = cv2.resize(image, new_size, self.interpolation_method)
        return image, new_size

    def resize_by_power(self, image: np.ndarray):
        """
        """
        height, width = image.shape[:2]
        # magnify image size
        target_size = min(max(height, width), 2048)
        ratio = target_size / max(height, width)
        self.input_size = int(height * ratio), int(width * ratio)
        image, _ = self.resize_no_aspect(image)
        # make canvas and paste image
        target_h32, target_w32 = image.shape[:2]
        if self.input_size[0] % 32 != 0:
            target_h32 = self.input_size[0] + (32 - self.input_size[0] % 32)
        if self.input_size[1] % 32 != 0:
            target_w32 = self.input_size[1] + (32 - self.input_size[1] % 32)

        self.input_size = (target_h32, target_w32)

        return image

    def run(self,
            image: np.ndarray) -> ImageResizerResult:
        """
        """

        if self.resize_to_power:
            resized_image = self.resize_by_power(image)

        else:
            if self.aspect_ration:
                resized_image, (resized_width, resized_height) = self.resize_with_aspect(image)

            else:
                resized_image, (resized_width, resized_height) = self.resize_no_aspect(image)

        if self.padding_type or self.resize_to_power:
            resized_image, (resized_width, resized_height) = self.resize_with_padding(resized_image)

        # calculate multiply coefficients
        height, width = image.shape[:2]
        kw, kh = width / resized_width, height / resized_height
        # get coords for slicing
        (y1, y2), (x1, x2) = self.get_new_coords(resized_width, resized_height)

        return ImageResizerResult(
            image=resized_image,
            coords=(x1, y1, x2, y2),
            scale=(kw, kh),
            height=height,
            width=width
        )