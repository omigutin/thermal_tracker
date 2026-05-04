"""Базовый класс для фильтрации ложных целей."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame


class BaseTargetFilter(ABC):
    """Фильтр принимает список объектов и возвращает очищенный список."""

    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def filter(
        self,
        frame: ProcessedFrame,
        objects: list[DetectedObject],
        motion: GlobalMotion,
    ) -> list[DetectedObject]:
        """Возвращает объекты, пережившие фильтрацию."""
