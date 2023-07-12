from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum
from math import isnan

import numpy as np

Point = Tuple[int, int]
Quadrangle = Tuple[Point, Point, Point, Point]  # (Left, Top), (Right, Top), (Right, Down), (Left, Down)
Column = List[Quadrangle]
RectBox = Tuple[int, int, int, int]


class RotationCls(str, Enum):
    deg0 = 'deg0'
    deg90 = 'deg90'
    deg180 = 'deg180'
    deg270 = 'deg270'


@dataclass
class ImageResizerResult:
    image: np.ndarray
    coords: RectBox
    scale: Tuple[float, float]
    height: int
    width: int


@dataclass
class SegmentationMaskResult:
    prediction_labels: np.ndarray
    coords: RectBox
    scale: Tuple[float, float]


@dataclass
class FieldMask:
    coords: Quadrangle
    rotation_cls: RotationCls
    confidence: float


@dataclass
class ModelPreprocessorResultProxy:
    image: np.ndarray
    coords: RectBox
    scale: Tuple[float, float]


@dataclass
class TextWithConfidence:
    text: str
    confidence: float

    def __post_init__(self):
        if isnan(self.confidence):
            raise ValueError(f'confidence is NaN for text with confidence: {self.text}')


class MultilangTranslateMode(str, Enum):
    ru = 'ru'
    en = 'en'
    mixed = 'mixed'
    passthrough = 'passthrough'
