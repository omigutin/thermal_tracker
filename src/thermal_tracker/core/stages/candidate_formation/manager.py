from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import ProcessedFrame
from ..motion_localization import MotionLocalizerResult
from .config import CandidateFormerConfig
from .factory import CandidateFormationFactory
from .operations import BaseCandidateFormer
from .result import CandidateFormerResult


class CandidateFormationManager:
    """Управляет выполнением операций формирования кандидатов."""

    def __init__(self, operations: Sequence[CandidateFormerConfig]) -> None:
        """Создать менеджер и подготовить активные runtime-операции."""
        self._operations: tuple[BaseCandidateFormer, ...] = CandidateFormationFactory.build_many(operations)

    @property
    def operations(self) -> tuple[BaseCandidateFormer, ...]:
        """Вернуть подготовленные runtime-операции формирования кандидатов."""
        return self._operations

    def apply(self,
        frame: ProcessedFrame,
        motion_localizer_result: MotionLocalizerResult,
    ) -> tuple[CandidateFormerResult, ...]:
        """Сформировать кандидатов по результату локализации движения."""
        candidates: list[CandidateFormerResult] = []

        for operation in self._operations:
            candidates.extend(operation.apply(frame=frame, motion_localizer_result=motion_localizer_result))

        return tuple(candidates)