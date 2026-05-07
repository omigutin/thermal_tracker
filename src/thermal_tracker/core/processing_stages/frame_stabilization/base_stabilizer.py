"""Базовый класс для оценки движения камеры."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import GlobalMotion, ProcessedFrame


class BaseMotionEstimator(ABC):
    """Любая стабилизация должна уметь оценить хотя бы грубый сдвиг кадра."""

    @abstractmethod
    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        """Возвращает оценку движения камеры между соседними кадрами."""
