"""Нормализация кадра по процентилям.

Часто полезна для тепловизора, когда несколько экстремальных пикселей
портят общий контраст и обычный min-max даёт слишком странную картинку.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .opencv_frame_preprocessing_utils import build_gradient, resize_if_needed, to_gray


class PercentileNormalizePreprocessor(BaseFramePreprocessor):
    """Сжимает хвосты распределения и растягивает рабочий диапазон."""

    def __init__(self, resize_width: int | None = 960, low_percentile: float = 2.0, high_percentile: float = 98.0) -> None:
        self.resize_width = resize_width
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        bgr = resize_if_needed(frame, self.resize_width)
        gray = to_gray(bgr)

        low = float(np.percentile(gray, self.low_percentile))
        high = float(np.percentile(gray, self.high_percentile))
        if high <= low:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            clipped = np.clip(gray.astype(np.float32), low, high)
            normalized = ((clipped - low) * (255.0 / (high - low))).astype(np.uint8)

        gradient = build_gradient(normalized, blur_kernel=3)
        return ProcessedFrame(bgr=bgr, gray=gray, normalized=normalized, gradient=gradient)
