"""Базовый класс для повторного захвата потерянной цели."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame


class BaseReacquirer(ABC):
    """Пытается вернуть цель после потери."""

    @abstractmethod
    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
    ) -> BoundingBox | None:
        """Возвращает новый bbox или `None`, если вернуть цель не удалось."""
