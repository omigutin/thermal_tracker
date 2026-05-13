"""Детектор движения на основе MOG2."""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import GlobalMotion, MotionDetectionResult, ProcessedFrame
from .base_moving_area_detector import BaseMotionDetector


class Mog2MotionDetector(BaseMotionDetector):
    """Использует `cv2.createBackgroundSubtractorMOG2`."""

    def __init__(self, history: int = 300, var_threshold: float = 20.0, detect_shadows: bool = False) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def detect(self, frame: ProcessedFrame, motion: GlobalMotion) -> MotionDetectionResult:
        raw_mask = self._subtractor.apply(frame.normalized)
        _, mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        score = float(np.count_nonzero(mask)) / max(mask.size, 1)
        return MotionDetectionResult(mask=mask, confidence_map=raw_mask, motion_score=score)
