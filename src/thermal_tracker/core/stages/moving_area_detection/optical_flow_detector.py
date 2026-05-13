"""Заготовка под детектор движения по optical flow."""

from __future__ import annotations

from .base_moving_area_detector import BaseMotionDetector


class OpticalFlowMotionDetector(BaseMotionDetector):
    """Будущий детектор движения на основе поля оптического потока."""

    def detect(self, frame, motion):
        raise NotImplementedError("Optical-flow детектор пока не реализован.")
