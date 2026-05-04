"""Детектор движения на основе бегущего среднего."""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import GlobalMotion, MotionDetectionResult, ProcessedFrame
from .base_moving_area_detector import BaseMotionDetector


class RunningAverageMotionDetector(BaseMotionDetector):
    """Сравнивает текущий кадр с медленно обновляемой моделью фона."""

    implementation_name = "running_average"
    is_ready = True

    def __init__(self, alpha: float = 0.04, threshold: int = 24) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self._background: np.ndarray | None = None

    def detect(self, frame: ProcessedFrame, motion: GlobalMotion) -> MotionDetectionResult:
        current = frame.normalized.astype(np.float32)
        if self._background is None:
            self._background = current.copy()
            return MotionDetectionResult(mask=np.zeros_like(frame.normalized), source_name=self.implementation_name)

        cv2.accumulateWeighted(current, self._background, self.alpha)
        background = cv2.convertScaleAbs(self._background)
        difference = cv2.absdiff(frame.normalized, background)
        _, mask = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        score = float(np.count_nonzero(mask)) / max(mask.size, 1)
        return MotionDetectionResult(mask=mask, confidence_map=difference, source_name=self.implementation_name, motion_score=score)
