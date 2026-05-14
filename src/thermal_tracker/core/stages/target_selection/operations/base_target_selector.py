from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import BoundingBox, ProcessedFrame
from ..result import TargetSelectorResult


class BaseTargetSelector(ABC):
    """Базовый интерфейс операции выбора цели."""

    @abstractmethod
    def apply(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> TargetSelectorResult:
        """Выбрать цель около точки клика или уточнить уже известную область."""
        raise NotImplementedError

    def refine(self, frame: ProcessedFrame, bbox: BoundingBox) -> TargetSelectorResult | None:
        """
            Уточнить уже выбранную область цели.
            По умолчанию повторно применяет selector в центре текущего bbox.
            Конкретная операция может переопределить метод и добавить свои проверки.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support target refinement.")
