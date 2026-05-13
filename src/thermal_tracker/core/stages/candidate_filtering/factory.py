from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NoReturn

from .config import CandidateFilterConfig
from .operations.area_aspect_candidate_filter import AreaAspectCandidateFilter, AreaAspectCandidateFilterConfig
from .operations.base_candidate_filter import BaseCandidateFilter
from .operations.border_touch_candidate_filter import BorderTouchCandidateFilter, BorderTouchCandidateFilterConfig
from .operations.contrast_candidate_filter import ContrastCandidateFilter, ContrastCandidateFilterConfig


# Связь конкретного класса конфигурации с runtime-классом фильтра.
# Источник правды для фабрики: добавление новой операции делается ровно здесь.
_RUNTIME_CLASSES: Mapping[type[CandidateFilterConfig], type[BaseCandidateFilter]] = {
    AreaAspectCandidateFilterConfig: AreaAspectCandidateFilter,
    BorderTouchCandidateFilterConfig: BorderTouchCandidateFilter,
    ContrastCandidateFilterConfig: ContrastCandidateFilter,
}


class CandidateFilterFactory:
    """
        Фабрика фильтров кандидатов.
        Создаёт готовые runtime-фильтры из объектов конфигурации, уже прошедших
        валидацию на этапе парсинга пресета.
        Фабрика не знает TOML, не работает с сырыми словарями и не парсит
        параметры операций. Связь `config -> runtime` хранится в одном
        приватном mapping модуля, что исключает дрейф между местами.
    """

    @classmethod
    def build_many(cls, operations: Sequence[CandidateFilterConfig]) -> tuple[BaseCandidateFilter, ...]:
        """Создать набор runtime-фильтров для активных конфигураций операций."""
        result: list[BaseCandidateFilter] = []

        for operation_config in operations:
            built_filter = cls.build(operation_config)
            if built_filter is not None:
                result.append(built_filter)

        return tuple(result)

    @classmethod
    def build(cls, operation_config: CandidateFilterConfig) -> BaseCandidateFilter | None:
        """Создать один runtime-фильтр из конфигурации операции."""
        if not operation_config.enabled:
            return None
        return cls._build_runtime_operation(operation_config)

    @classmethod
    def _build_runtime_operation(cls, operation_config: CandidateFilterConfig) -> BaseCandidateFilter:
        """Создать runtime-фильтр по точному типу конфигурации."""
        runtime_class = _RUNTIME_CLASSES.get(type(operation_config))
        if runtime_class is None:
            cls._raise_invalid_config(operation_config)
        return runtime_class(config=operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации фильтра."""
        raise TypeError(f"Unsupported candidate filter config: {type(operation_config).__name__!r}.")
