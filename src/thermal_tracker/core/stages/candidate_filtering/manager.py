from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .config import CandidateFilterConfig
from .factory import CandidateFilterFactory
from .operations.base_candidate_filter import BaseCandidateFilter


class CandidateFilterManager:
    """
        Менеджер стадии фильтрации кандидатов.
        Принимает упорядоченный набор конфигураций операций, делегирует их
        сборку фабрике и последовательно применяет полученные runtime-фильтры
        к списку кандидатов.
        Менеджер не знает деталей TOML-пресета, не парсит сырые значения и
        не создаёт фильтры вручную: всё это лежит на парсере пресета и фабрике.
    """

    def __init__(self, operations: Sequence[CandidateFilterConfig]) -> None:
        """Подготовить runtime-фильтры из конфигураций операций."""
        self._operations: tuple[BaseCandidateFilter, ...] = (CandidateFilterFactory.build_many(operations))

    @property
    def operations(self) -> tuple[BaseCandidateFilter, ...]:
        """Вернуть подготовленные runtime-фильтры в порядке применения."""
        return self._operations

    def apply(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        """Последовательно применить runtime-фильтры к списку кандидатов."""

        current = list(objects)
        for runtime_operation in self._operations:
            current = runtime_operation.filter(frame=frame, objects=current, motion=motion)

        return current
