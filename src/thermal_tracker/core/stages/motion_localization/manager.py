from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import ProcessedFrame
from .config import MotionLocalizationConfig
from .factory import MotionLocalizationFactory
from .operations.base_motion_localizer import BaseMotionLocalizer
from .result import MotionLocalizerResult


class MotionLocalizationManager:
    """Управляет выполнением операции локализации движения на кадре."""

    def __init__(self, operations: Sequence[MotionLocalizationConfig]) -> None:
        """Создать менеджер и подготовить активные runtime-операции."""
        self._operations: tuple[BaseMotionLocalizer, ...] = (MotionLocalizationFactory.build_many(operations))
        self._validate_operations_count()

    @property
    def operations(self) -> tuple[BaseMotionLocalizer, ...]:
        """Вернуть подготовленные runtime-операции локализации движения."""
        return self._operations

    def localize(self, frame: ProcessedFrame) -> MotionLocalizerResult:
        """Вернуть результат локализации движения на кадре."""
        if not self._operations:
            return MotionLocalizerResult.empty_like(frame.normalized)
        return self._operations[0].apply(frame)

    def _validate_operations_count(self) -> None:
        """Проверить, что активна не более одной операции локализации движения."""
        if len(self._operations) > 1:
            raise ValueError(
                "Motion localization supports only one active operation. "
                "Result merging for multiple operations is not implemented."
            )
