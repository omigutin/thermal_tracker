"""Атомарная операция: гауссово сглаживание серого канала."""

from __future__ import annotations

from dataclasses import replace

import cv2

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .opencv_frame_preprocessing_utils import make_odd


class GaussianBlurFramePreprocessor(BaseFramePreprocessor):
    """Сглаживает gray-канал гауссовым фильтром."""

    def __init__(self, kernel: int) -> None:
        self._kernel = make_odd(max(1, kernel))

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        if self._kernel <= 1:
            return frame
        gray = cv2.GaussianBlur(frame.gray, (self._kernel, self._kernel), 0)
        return replace(frame, gray=gray)
