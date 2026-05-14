from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import ProcessedFrame
from .config import FrameStabilizerConfig
from .factory import FrameStabilizerFactory
from .operations import BaseFrameStabilizer
from .result import FrameStabilizerResult


class FrameStabilizerManager:
    """Управляет выполнением операции стабилизации кадра."""

    def __init__(self, operations: Sequence[FrameStabilizerConfig]) -> None:
        """Создать менеджер и подготовить активные runtime-операции."""
        self._operations: tuple[BaseFrameStabilizer, ...] = FrameStabilizerFactory.build_many(operations)
        self._validate_operations_count()

    @property
    def operations(self) -> tuple[BaseFrameStabilizer, ...]:
        """Вернуть подготовленные runtime-операции стабилизации кадра."""
        return self._operations

    def apply(self, frame: ProcessedFrame) -> FrameStabilizerResult:
        """Вернуть результат стабилизации текущего кадра."""
        if not self._operations:
            return FrameStabilizerResult()
        return self._operations[0].apply(frame)

    def _validate_operations_count(self) -> None:
        """Проверить, что активна не более одной операции стабилизации кадра."""
        if len(self._operations) > 1:
            raise ValueError(
                "Frame stabilization supports only one active operation. "
                "Result merging for multiple operations is not implemented."
            )
