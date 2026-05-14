"""Типы методов повторного захвата цели."""

from __future__ import annotations

from enum import StrEnum


class TargetRecovererType(StrEnum):
    """Доступные методы стадии target_recovery."""

    LOCAL_TEMPLATE = "local_template"  # Ищет потерянную цель рядом с последним bbox по шаблону.
    GLOBAL_SEARCH = "global_search"  # Расширяет поиск до большой области или всего кадра.
    CANDIDATE_BASED = "candidate_based"  # Пытается вернуть цель через список новых кандидатов.
    MULTI_SCALE = "multi_scale"  # Ищет цель по шаблону с перебором масштабов.
    IRST_CONTRAST = "irst_contrast"  # IRST: локальный контраст в расширенной зоне, без шаблонов.
