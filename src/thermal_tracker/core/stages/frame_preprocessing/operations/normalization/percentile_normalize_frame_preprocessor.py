"""Атомарная операция: перцентильная нормализация gray в normalized.

Полезна для тепловизора: пара экстремальных пикселей не растягивает диапазон,
а отрезается на заданных перцентилях.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .....domain.models import ProcessedFrame
from ..base_frame_preprocessor import BaseFramePreprocessor


class PercentileNormalizePreprocessor(BaseFramePreprocessor):
    """Сжимает хвосты распределения яркости и пишет результат в normalized."""

    def __init__(self, low_percentile: float = 2.0, high_percentile: float = 98.0) -> None:
        self._low = low_percentile
        self._high = high_percentile

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        gray = frame.gray
        low = float(np.percentile(gray, self._low))
        high = float(np.percentile(gray, self._high))
        if high <= low:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            clipped = np.clip(gray.astype(np.float32), low, high)
            normalized = ((clipped - low) * (255.0 / (high - low))).astype(np.uint8)
        return replace(frame, normalized=normalized)
