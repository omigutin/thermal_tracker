from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import CandidateFilterConfig, CANDIDATE_FILTER_CONFIG_CLASSES
from .operations.area_aspect_candidate_filter import AreaAspectCandidateFilter, AreaAspectCandidateFilterConfig
from .operations.base_candidate_filter import BaseCandidateFilter
from .operations.border_touch_candidate_filter import BorderTouchCandidateFilter, BorderTouchCandidateFilterConfig
from .operations.contrast_candidate_filter import ContrastCandidateFilter, ContrastCandidateFilterConfig


class CandidateFilterFactory:
    """
        Фабрика фильтров кандидатов.
        Модуль отвечает за создание готовых объектов фильтров из описаний,
        которые приходят из конфигурации, пресета или внутреннего кода.
        Фабрика отделяет создание фильтров от менеджера. Благодаря этому
        `CandidateFilterManager` занимается только последовательным применением уже подготовленных фильтров.
    """

    @classmethod
    def build_many(cls, operations: Sequence[CandidateFilterConfig]) -> tuple[BaseCandidateFilter, ...]:
        """Создать набор активных фильтров в исходном порядке."""

        result: list[BaseCandidateFilter] = []

        for operation in operations:
            built_filter = cls.build(operation)
            if built_filter is not None:
                result.append(built_filter)

        return tuple(result)

    @classmethod
    def build(cls, operation_config: CandidateFilterConfig) -> BaseCandidateFilter | None:
        """Создать один runtime-фильтр из конфигурации фильтра."""
        cls._validate_operation_config(operation_config)
        if not operation_config.enabled:
            return None
        return cls._build_runtime_filter(operation_config)

    @classmethod
    def _build_runtime_filter(cls, operation_config: CandidateFilterConfig) -> BaseCandidateFilter:
        """Создать runtime-фильтр по конкретному типу конфигурации."""
        if isinstance(operation_config, AreaAspectCandidateFilterConfig):
            return AreaAspectCandidateFilter(config=operation_config)
        if isinstance(operation_config, BorderTouchCandidateFilterConfig):
            return BorderTouchCandidateFilter(config=operation_config)
        if isinstance(operation_config, ContrastCandidateFilterConfig):
            return ContrastCandidateFilter(config=operation_config)
        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: CandidateFilterConfig) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(operation_config, tuple(CANDIDATE_FILTER_CONFIG_CLASSES.values())):
            return
        CandidateFilterFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации фильтра."""
        raise TypeError(f"Unsupported candidate filter config: {type(operation_config).__name__!r}.")
