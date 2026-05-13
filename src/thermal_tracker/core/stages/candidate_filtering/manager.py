"""
    Менеджер фильтров кандидатов.
    Класс отвечает за последовательный запуск одного или нескольких фильтров
    кандидатов на цель.

    Основная задача менеджера:
        1. Принять описание фильтров при инициализации.
        2. Передать создание фильтров в CandidateFilterFactory.
        3. Хранить готовые фильтры в неизменяемом порядке.
        4. Последовательно применять фильтры к списку обнаруженных объектов.

    Поддерживаемые варианты входных фильтров:

        1. CandidateFilterType:
            Менеджер создаст фильтр по enum-значению.

            CandidateFilterManager(
                operations=(
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
                operations=(
                    "area_aspect",
                    "border_touch",
                    "contrast",
                ),
            )

        3. CandidateFilterConfig:
            Менеджер создаст фильтр по описанию из пресета.
            Если enabled=False, фильтр не попадёт в runtime pipeline.

            CandidateFilterManager(
                operations=(
                    CandidateFilterConfig(
                        type=CandidateFilterType.CONTRAST,
                        enabled=True,
                        params={
                            "min_contrast": 5.0,
                            "border": 6,
                        },
                    ),
                ),
            )
"""

from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .config import CandidateFilterConfig
from .factory import CandidateFilterFactory
from .operations.base_candidate_filter import BaseCandidateFilter


class CandidateFilterManager:
    """Менеджер атомарных фильтров для отсеивания кандидатов."""

    def __init__(self, filters: Sequence[CandidateFilterConfig], ) -> None:
        """Инициализировать менеджер и подготовить фильтры к запуску."""
        self._filters: tuple[BaseCandidateFilter, ...] = (CandidateFilterFactory.build_many(filters))

    @property
    def filters(self) -> tuple[BaseCandidateFilter, ...]:
        """Вернуть подготовленные экземпляры фильтров."""
        return self._filters

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion, ) -> list[DetectedObject]:
        """Последовательно применить фильтры к кандидатам."""

        current = list(objects)

        for candidate_filter in self._filters:
            current = candidate_filter.filter(frame=frame, objects=current, motion=motion, )

        return current