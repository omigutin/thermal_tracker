from __future__ import annotations

from enum import StrEnum


class CandidateFormationType(StrEnum):
    """Типы операций формирования кандидатов."""

    CONNECTED_COMPONENTS = "connected_components"   # Собирает кандидатов из компонент связности маски
    CONTOUR = "contour"  # Собирает кандидатов по внешним контурам маски
