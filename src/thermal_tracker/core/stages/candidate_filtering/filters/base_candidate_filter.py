"""
    Базовый интерфейс фильтра кандидатов.
    Модуль содержит абстрактный класс для всех фильтров, которые отбрасывают ложные или нежелательные объекты-кандидаты.

    Каждый конкретный фильтр принимает:
        - обработанный кадр;
        - текущий список объектов-кандидатов;
        - информацию о глобальном движении кадра.

    На выходе фильтр возвращает новый список объектов, которые прошли проверку.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import DetectedObject, GlobalMotion, ProcessedFrame


class BaseCandidateFilter(ABC):
    """ Базовый интерфейс атомарного фильтра кандидатов. """

    @abstractmethod
    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion,) -> list[DetectedObject]:
        """ Отфильтровать список объектов-кандидатов. """
        pass
