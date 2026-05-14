from __future__ import annotations

from enum import StrEnum


class CandidateFilterType(StrEnum):
    """
        Типы атомарных фильтров кандидатов.
        Модуль содержит enum со всеми фильтрами, которые могут быть использованы в стадии candidate_filtering.
        Enum нужен для конфигураций и пресетов: внешний код может передать тип фильтра в CandidateFilterManager,
        а менеджер преобразует его в готовый экземпляр соответствующего фильтра.
    """
    AREA_ASPECT = "area_aspect"  # Отсекает кандидатов по площади и пропорциям рамки.
    BORDER_TOUCH = "border_touch"  # Убирает кандидатов, которые касаются края кадра.
    CONTRAST = "contrast"  # Проверяет, что кандидат заметно отличается от локального фона.
