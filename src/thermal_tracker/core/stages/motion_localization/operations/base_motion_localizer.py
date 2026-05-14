from __future__ import annotations

from abc import ABC, abstractmethod

from ..result import MotionLocalizerResult
from ....domain.models import ProcessedFrame


class BaseMotionLocalizerConfig:
    """Базовый класс конфигураций локализации движения."""

    @staticmethod
    def validate_odd_positive_kernel(value: int, field_name: str) -> None:
        """Проверить, что размер ядра положительный и нечётный."""
        if value < 1:
            raise ValueError(f"{field_name} must be greater than or equal to 1.")
        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd.")


class BaseMotionLocalizer(ABC):
    """Базовый интерфейс операции локализации движения."""

    @abstractmethod
    def apply(self, frame: ProcessedFrame) -> MotionLocalizerResult:
        """Возвращает результат обнаружения движения на кадре."""
        raise NotImplementedError