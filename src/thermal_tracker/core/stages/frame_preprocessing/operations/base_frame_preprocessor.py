from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import ProcessedFrame


class BaseFramePreprocessor(ABC):
    """Базовый интерфейс операции предобработки кадра."""

    @abstractmethod
    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить операцию к кадру и вернуть обновлённый ProcessedFrame."""
        raise NotImplementedError
