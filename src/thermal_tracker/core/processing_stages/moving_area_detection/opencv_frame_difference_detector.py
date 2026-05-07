"""Детектор движения по разности соседних кадров."""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import GlobalMotion, MotionDetectionResult, ProcessedFrame
from .base_moving_area_detector import BaseMotionDetector


class FrameDifferenceMotionDetector(BaseMotionDetector):
    """Ищет движение по абсолютной разности нормализованных кадров."""

    implementation_name = "frame_difference"
    is_ready = True

    def __init__(self, threshold: int = 22, blur_kernel: int = 5) -> None:
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self._previous: np.ndarray | None = None

    def detect(self, frame: ProcessedFrame, motion: GlobalMotion) -> MotionDetectionResult:
        current = frame.normalized
        if self._previous is None:
            self._previous = current.copy()
            return MotionDetectionResult(mask=np.zeros_like(current), source_name=self.implementation_name)

        difference = cv2.absdiff(current, self._previous)
        self._previous = current.copy()

        if self.blur_kernel > 1:
            difference = cv2.GaussianBlur(difference, (self.blur_kernel, self.blur_kernel), 0)

        _, mask = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        score = float(np.count_nonzero(mask)) / max(mask.size, 1)
        return MotionDetectionResult(mask=mask, confidence_map=difference, source_name=self.implementation_name, motion_score=score)
