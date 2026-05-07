"""Самый честный препроцессор: почти ничего не делает.

Полезен как baseline:
- позволяет понять, помогает ли нам сложная предобработка вообще;
- полезен для отладки, когда надо исключить лишнюю магию.
"""

from __future__ import annotations

import numpy as np

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .opencv_frame_preprocessing_utils import build_gradient, normalize_minmax, resize_if_needed, to_gray


class IdentityFramePreprocessor(BaseFramePreprocessor):
    """Возвращает кадр почти как есть."""

    def __init__(self, resize_width: int | None = 960) -> None:
        self.resize_width = resize_width

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        bgr = resize_if_needed(frame, self.resize_width)
        gray = to_gray(bgr)
        normalized = normalize_minmax(gray)
        gradient = build_gradient(normalized, blur_kernel=1)
        return ProcessedFrame(bgr=bgr, gray=gray, normalized=normalized, gradient=gradient)
