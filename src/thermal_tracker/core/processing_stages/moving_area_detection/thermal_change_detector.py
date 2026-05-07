"""Заготовка под thermal-aware детектор изменений."""

from __future__ import annotations

from .base_moving_area_detector import BaseMotionDetector


class ThermalChangeMotionDetector(BaseMotionDetector):
    """Будущий детектор изменений, более ориентированный на thermal-сцены."""

    implementation_name = "thermal_change"
    is_ready = False

    def detect(self, frame, motion):
        raise NotImplementedError("Thermal-aware детектор пока не реализован.")
