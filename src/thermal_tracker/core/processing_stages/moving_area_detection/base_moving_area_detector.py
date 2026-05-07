"""Базовый класс для детекторов движения."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import GlobalMotion, MotionDetectionResult, ProcessedFrame


class BaseMotionDetector(ABC):
    """Любой детектор движения должен вернуть хотя бы бинарную маску."""

    @abstractmethod
    def detect(self, frame: ProcessedFrame, motion: GlobalMotion) -> MotionDetectionResult:
        """Возвращает результат обнаружения движения на кадре."""
