"""Базовый контракт атомарной операции стадии preprocessing.

Каждая реализация изменяет один или несколько каналов :class:`ProcessedFrame`
(``bgr``, ``gray``, ``normalized``, ``gradient``, ``quality``) и возвращает
обновлённый кадр. Менеджер последовательно прогоняет ProcessedFrame через
несколько таких операций.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import ProcessedFrame


class BaseFramePreprocessor(ABC):
    """Атомарная операция предобработки кадра."""

    @abstractmethod
    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить операцию и вернуть обновлённый ProcessedFrame."""
