"""Предобработка с акцентом на локальный контраст.

Это хороший вариант, когда цель видна, но сцена плоская и невыразительная.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .opencv_frame_preprocessing_utils import build_gradient, normalize_minmax, resize_if_needed, to_gray


class ClaheContrastPreprocessor(BaseFramePreprocessor):
    """Усиливает локальный контраст через CLAHE."""

    implementation_name = "clahe_contrast"
    is_ready = True

    def __init__(self, resize_width: int | None = 960, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        self.resize_width = resize_width
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        bgr = resize_if_needed(frame, self.resize_width)
        gray = to_gray(bgr)
        normalized = normalize_minmax(gray)
        normalized = self._clahe.apply(normalized)
        gradient = build_gradient(normalized, blur_kernel=3)
        return ProcessedFrame(bgr=bgr, gray=gray, normalized=normalized, gradient=gradient)
