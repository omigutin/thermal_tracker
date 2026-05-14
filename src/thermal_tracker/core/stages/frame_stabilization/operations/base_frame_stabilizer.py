from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import ProcessedFrame
from ..result import FrameStabilizerResult


class BaseFrameStabilizer(ABC):
    """Базовый интерфейс операции стабилизации кадра."""

    @abstractmethod
    def apply(self, frame: ProcessedFrame) -> FrameStabilizerResult:
        """Вернуть результат стабилизации текущего кадра."""
        raise NotImplementedError
