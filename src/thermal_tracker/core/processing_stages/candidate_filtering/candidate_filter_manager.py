"""
    Менеджер фильтров кандидатов.
    Класс отвечает за последовательный запуск одного или нескольких фильтров
    кандидатов на цель.
    Основная задача менеджера:
        1. Принять описание фильтров при инициализации.
        2. Один раз привести входные значения к готовым экземплярам фильтров.
        3. Хранить фильтры в неизменяемом порядке.
        4. Последовательно применять фильтры к списку обнаруженных объектов.

    Поддерживаемые варианты входных фильтров:

        1. CandidateFilterType:
            Менеджер создаст фильтр по enum-значению.

            CandidateFilterManager(
                filters=(
                    CandidateFilterType.AREA_ASPECT,
                    CandidateFilterType.BORDER_TOUCH,
                    CandidateFilterType.CONTRAST,
                ),
            )

        2. str:
            Менеджер преобразует строку в CandidateFilterType, а затем создаст
            соответствующий фильтр.

            Поддерживаются строки по value:
                "area_aspect"
                "border_touch"
                "contrast"

            Также поддерживаются строки по name:
                "AREA_ASPECT"
                "BORDER_TOUCH"
                "CONTRAST"

            CandidateFilterManager(
                filters=(
                    "area_aspect",
                    "border_touch",
                    "contrast",
                ),
            )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .area_aspect_candidate_filter import AreaAspectCandidateFilter
from .base_candidate_filter import BaseCandidateFilter
from .border_touch_candidate_filter import BorderTouchCandidateFilter
from .candidate_filter_type import CandidateFilterType
from .contrast_candidate_filter import ContrastCandidateFilter


CandidateFilterInput: TypeAlias = CandidateFilterType | str


class CandidateFilterManager:
    """Менеджер атомарных фильтров для отсеивания кандидатов."""

    _filter_classes: dict[
        CandidateFilterType,
        type[BaseCandidateFilter],
    ] = {
        CandidateFilterType.AREA_ASPECT: AreaAspectCandidateFilter,
        CandidateFilterType.BORDER_TOUCH: BorderTouchCandidateFilter,
        CandidateFilterType.CONTRAST: ContrastCandidateFilter,
    }

    def __init__(
        self,
        filters: Sequence[CandidateFilterInput],
    ) -> None:
        """Инициализировать менеджер и подготовить фильтры к запуску."""

        self._filters: tuple[BaseCandidateFilter, ...] = (
            self._normalize_filters(filters)
        )

    @property
    def filters(self) -> tuple[BaseCandidateFilter, ...]:
        """Вернуть подготовленные экземпляры фильтров."""

        return self._filters

    @classmethod
    def _normalize_filters(
        cls,
        filters: Sequence[CandidateFilterInput],
    ) -> tuple[BaseCandidateFilter, ...]:
        """Преобразовать входные описания фильтров в экземпляры фильтров."""

        return tuple(
            cls._normalize_filter(candidate_filter)
            for candidate_filter in filters
        )

    @classmethod
    def _normalize_filter(
        cls,
        candidate_filter: CandidateFilterInput,
    ) -> BaseCandidateFilter:
        """Преобразовать одно описание фильтра в экземпляр фильтра."""

        filter_type = cls._normalize_filter_type(candidate_filter)
        filter_cls = cls._filter_classes.get(filter_type)

        if filter_cls is None:
            raise ValueError(
                f"Unsupported candidate filter type: {filter_type!r}. "
                f"Available filter types: {tuple(cls._filter_classes)}."
            )

        return filter_cls()

    @staticmethod
    def _normalize_filter_type(candidate_filter: CandidateFilterType | str, ) -> CandidateFilterType:
        """Преобразовать строку или enum-значение в CandidateFilterType."""

        if isinstance(candidate_filter, CandidateFilterType):
            return candidate_filter

        try:
            return CandidateFilterType(candidate_filter)
        except ValueError:
            pass

        filter_by_name = CandidateFilterType.__members__.get(candidate_filter.upper())
        if filter_by_name is not None:
            return filter_by_name

        raise ValueError(
            f"Unsupported candidate filter value: {candidate_filter!r}. "
            f"Available filter values: "
            f"{tuple(item.value for item in CandidateFilterType)}. "
            f"Available filter names: "
            f"{tuple(item.name for item in CandidateFilterType)}."
        )

    def filter(
        self,
        frame: ProcessedFrame,
        objects: list[DetectedObject],
        motion: GlobalMotion,
    ) -> list[DetectedObject]:
        """Последовательно применить фильтры к кандидатам."""

        current = list(objects)

        for candidate_filter in self._filters:
            current = candidate_filter.filter(
                frame=frame,
                objects=current,
                motion=motion,
            )

        return current