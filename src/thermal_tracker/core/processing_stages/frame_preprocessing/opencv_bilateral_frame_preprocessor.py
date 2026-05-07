"""Предобработка через bilateral filter.

Нужна там, где хочется приглушить шум, но по возможности не размазать границы.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .opencv_frame_preprocessing_utils import build_gradient, normalize_minmax, resize_if_needed, to_gray


class BilateralFramePreprocessor(BaseFramePreprocessor):
    """Сглаживает шум, стараясь сохранить границы объекта."""

    implementation_name = "bilateral"
    is_ready = True

    def __init__(self, resize_width: int | None = 960, diameter: int = 7, sigma_color: float = 40.0, sigma_space: float = 40.0) -> None:
        self.resize_width = resize_width
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        bgr = resize_if_needed(frame, self.resize_width)
        gray = to_gray(bgr)
        filtered = cv2.bilateralFilter(gray, self.diameter, self.sigma_color, self.sigma_space)
        normalized = normalize_minmax(filtered)
        gradient = build_gradient(normalized, blur_kernel=3)
        return ProcessedFrame(bgr=bgr, gray=filtered, normalized=normalized, gradient=gradient)
