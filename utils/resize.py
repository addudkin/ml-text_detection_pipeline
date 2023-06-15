import cv2
import random
import numpy as np

from typing import Tuple


class ImageResizer(object):
    def __init__(self,
                 input_size: Tuple[int, int],
                 aspect_ration: bool = True,
                 padding_type: str = "",
                 padding_value: int = 0,
                 interpolation_method: int = cv2.INTER_NEAREST) -> None:

        self.input_size = input_size  # height, width
        self.aspect_ration = aspect_ration
        self.interpolation_method = interpolation_method
        self.padding_value = padding_value
        self.padding_type = padding_type.upper()

        assert self.padding_type in ("RIGHT", "LEFT", "BOTH", ""), "Incorrect padding type"

    def get_new_size(self,
                     image: np.ndarray) -> Tuple[int, int]:

        img_h, img_w = image.shape[:2]
        new_h, new_w = self.input_size

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
            x2, y2 = new_h, new_w

        return (y1, y2), (x1, x2)

    def resize_with_padding(self,
                            image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        """

        image, (resized_width, resized_height) = self.resize_with_aspect(image)
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

    def resize(self,
               image: np.ndarray) -> Tuple[np.ndarray,
                                           Tuple[Tuple[int, int], Tuple[int, int]],
                                           Tuple[float, float]]:
        """
        """
        if not self.aspect_ration:
            resized_image, (resized_width, resized_height) = self.resize_no_aspect(image)

        else:
            if self.padding_type:
                resized_image, (resized_width, resized_height) = self.resize_with_padding(image)
            else:
                resized_image, (resized_width, resized_height) = self.resize_with_aspect(image)

        # calculate multiply coefficients
        height, width = image.shape[:2]
        kw, kh = width / resized_width, height / resized_height
        # get coords for slicing
        (y1, y2), (x1, x2) = self.get_new_coords(resized_width, resized_height)

        return resized_image, ((y1, y2), (x1, x2)), (kw, kh)


def resize(image: np.ndarray,
           mask: np.ndarray,
           img_size: Tuple[int, int],
           split: str) -> Tuple[np.ndarray, np.ndarray]:
    padding_value = 0
    padding_type = "BOTH"
    interpolation_method = cv2.INTER_NEAREST

    resizer = ImageResizer(img_size,
                           padding_value=padding_value,
                           interpolation_method=interpolation_method)

    if split == "TRAIN":
        interpolation_method = random.choice([cv2.INTER_AREA, cv2.INTER_BITS, cv2.INTER_BITS2,
                                              cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR,
                                              cv2.INTER_LINEAR_EXACT, cv2.INTER_MAX, cv2.INTER_NEAREST,
                                              cv2.INTER_NEAREST_EXACT, cv2.INTER_TAB_SIZE, cv2.INTER_TAB_SIZE2])
        padding_type = random.choice(["BOTH", "LEFT", "RIGHT"])
        padding_value = random.randint(0, 255)

    resizer.padding_type = padding_type

    mask = resizer.resize_with_padding(mask)[0]

    resizer.interpolation_method = interpolation_method
    resizer.padding_value = padding_value

    image = resizer.resize_with_padding(image)[0]

    return image, mask
