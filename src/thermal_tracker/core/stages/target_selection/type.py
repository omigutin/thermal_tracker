from __future__ import annotations

from enum import StrEnum


class TargetSelectorType(StrEnum):
    """Типы операций выбора цели."""
    # Выбирает цель вокруг клика по локальному контрасту и компонентам связности.
    CONTRAST_COMPONENT = "contrast_component"
    # Выбирает цель вокруг клика через GrabCut-сегментацию локальной области.
    GRABCUT = "grabcut"
