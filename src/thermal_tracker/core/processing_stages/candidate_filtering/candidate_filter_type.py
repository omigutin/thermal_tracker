"""
    Типы атомарных фильтров кандидатов.
    Модуль содержит enum со всеми фильтрами, которые могут быть использованы в стадии candidate_filtering.
    Enum нужен для конфигураций и пресетов: внешний код может передать тип фильтра в CandidateFilterManager,
    а менеджер преобразует его в готовый экземпляр соответствующего фильтра.
"""

from __future__ import annotations

from enum import StrEnum


class CandidateFilterType(StrEnum):
    """ Доступные типы атомарных фильтров кандидатов. """
    AREA_ASPECT = "area_aspect"
    BORDER_TOUCH = "border_touch"
    CONTRAST = "contrast"
