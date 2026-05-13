"""Нулевая оценка движения камеры."""

from __future__ import annotations

from ...domain.models import GlobalMotion, ProcessedFrame
from .base_stabilizer import BaseMotionEstimator


class NoMotionEstimator(BaseMotionEstimator):
    """Возвращает отсутствие сдвига, когда стабилизация кадра отключена."""

    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        """Возвращает пустую оценку движения камеры."""

        return GlobalMotion()
