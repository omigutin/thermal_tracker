from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import BoundingBox, ProcessedFrame
from .config import TargetSelectionConfig
from .factory import TargetSelectionFactory
from .operations import BaseTargetSelector
from .result import TargetSelectorResult


class TargetSelectionManager:
    """Управляет выполнением операции выбора цели."""

    def __init__(self, operations: Sequence[TargetSelectionConfig]) -> None:
        """Создать менеджер и подготовить активные runtime-операции."""
        self._operations: tuple[BaseTargetSelector, ...] = (TargetSelectionFactory.build_many(operations))
        self._validate_operations_count()

    @property
    def operations(self) -> tuple[BaseTargetSelector, ...]:
        """Вернуть подготовленные runtime-операции выбора цели."""
        return self._operations

    def apply(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> TargetSelectorResult | None:
        """Выбрать цель около точки клика."""
        if not self._operations:
            return None
        return self._operations[0].apply(frame=frame, point=point, expected_bbox=expected_bbox)

    def refine(self, frame: ProcessedFrame, bbox: BoundingBox) -> TargetSelectorResult | None:
        """Уточнить уже выбранную область цели."""
        if not self._operations:
            return None
        return self._operations[0].refine(frame=frame, bbox=bbox)

    def _validate_operations_count(self) -> None:
        """Проверить, что активна не более одной операции выбора цели."""
        if len(self._operations) > 1:
            raise ValueError(
                "Target selection supports only one active operation. "
                "Fallback chain for multiple selectors is not implemented."
            )
