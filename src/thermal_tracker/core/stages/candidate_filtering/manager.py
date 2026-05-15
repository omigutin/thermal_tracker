from __future__ import annotations

from thermal_tracker.core.stages.config.stage_config import StageConfig
from ...domain.models import ProcessedFrame
from ..candidate_formation.result import CandidateFormerResult
from ..frame_stabilization import FrameStabilizerResult
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

    def __init__(self, config: StageConfig[CandidateFilterConfig]) -> None:
        """Подготовить runtime-фильтры из конфигурации стадии."""
        # Конфигурация стадии фильтрации кандидатов.
        self._config: StageConfig[CandidateFilterConfig] = config
        # Подготовленные runtime-фильтры в порядке применения.
        self._operations: tuple[BaseCandidateFilter, ...] = self._build_operations(config)

    @property
    def config(self) -> StageConfig[CandidateFilterConfig]:
        """Вернуть конфигурацию стадии фильтрации кандидатов."""
        return self._config

    @property
    def operations(self) -> tuple[BaseCandidateFilter, ...]:
        """Вернуть подготовленные runtime-фильтры в порядке применения."""
        return self._operations

    def apply(
        self,
        frame: ProcessedFrame,
        objects: list[CandidateFormerResult],
        motion: FrameStabilizerResult,
    ) -> list[CandidateFormerResult]:
        """Последовательно применить runtime-фильтры к списку кандидатов."""
        if not self._config.enabled or not self._operations:
            return list(objects)

        current = list(objects)

        for runtime_operation in self._operations:
            current = runtime_operation.apply(frame=frame, objects=current, motion=motion)

        return current

    @staticmethod
    def _build_operations(config: StageConfig[CandidateFilterConfig]) -> tuple[BaseCandidateFilter, ...]:
        """Создать runtime-фильтры, если стадия включена."""
        if not config.enabled:
            return ()

        return CandidateFilterFactory.build_many(config.operations)
