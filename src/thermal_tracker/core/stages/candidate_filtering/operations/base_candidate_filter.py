from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import ProcessedFrame
from ...candidate_formation.result import CandidateFormerResult
from ...frame_stabilization.result import FrameStabilizerResult


class BaseCandidateFilter(ABC):
    """
        Базовый интерфейс фильтра кандидатов.
        Модуль содержит абстрактный класс для всех фильтров, которые отбрасывают ложные или нежелательные объекты-кандидаты.

        Каждый конкретный фильтр принимает:
            - обработанный кадр;
            - текущий список объектов-кандидатов;
            - информацию о глобальном движении кадра.

        На выходе фильтр возвращает новый список объектов, которые прошли проверку.
    """

    @abstractmethod
    def apply(self,
              frame: ProcessedFrame,
              objects: list[CandidateFormerResult],
              motion: FrameStabilizerResult,
              ) -> list[CandidateFormerResult]:
        """ Отфильтровать список объектов-кандидатов. """
        raise NotImplementedError
