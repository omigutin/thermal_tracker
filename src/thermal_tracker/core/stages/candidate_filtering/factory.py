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
    def build_many(cls, filters: Sequence[CandidateFilterConfig]) -> tuple[BaseCandidateFilter, ...]:
        """Создать набор активных фильтров в исходном порядке."""

        result: list[BaseCandidateFilter] = []

        for candidate_filter_config in filters:
            built_filter = cls.build(candidate_filter_config)
            if built_filter is not None:
                result.append(built_filter)

        return tuple(result)

    @classmethod
    def build(cls, filter_config: CandidateFilterConfig) -> BaseCandidateFilter | None:
        """Создать один runtime-фильтр из конфигурации фильтра."""
        cls._validate_filter_config(filter_config)
        if not filter_config.enabled:
            return None
        return cls._build_runtime_filter(filter_config)

    @classmethod
    def _build_runtime_filter(cls, filter_config: CandidateFilterConfig) -> BaseCandidateFilter:
        """Создать runtime-фильтр по конкретному типу конфигурации."""
        if isinstance(filter_config, AreaAspectCandidateFilterConfig):
            return AreaAspectCandidateFilter(config=filter_config)
        if isinstance(filter_config, BorderTouchCandidateFilterConfig):
            return BorderTouchCandidateFilter(config=filter_config)
        if isinstance(filter_config, ContrastCandidateFilterConfig):
            return ContrastCandidateFilter(config=filter_config)
        cls._raise_invalid_config(filter_config)

    @staticmethod
    def _validate_filter_config(filter_config: CandidateFilterConfig) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(filter_config, tuple(CANDIDATE_FILTER_CONFIG_CLASSES.values())):
            return
        CandidateFilterFactory._raise_invalid_config(filter_config)

    @staticmethod
    def _raise_invalid_config(filter_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации фильтра."""
        raise TypeError(f"Unsupported candidate filter config: {type(filter_config).__name__!r}.")
