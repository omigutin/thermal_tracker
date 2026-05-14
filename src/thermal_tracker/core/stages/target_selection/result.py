from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.models import BoundingBox


class TargetPolarity(StrEnum):
    """Тип яркости выбранной цели относительно окружения."""
    HOT = "hot"
    COLD = "cold"


@dataclass(slots=True)
class TargetSelectorResult:
    """Результат выбора цели."""
    bbox: BoundingBox
    confidence: float
    polarity: TargetPolarity

    def __post_init__(self) -> None:
        """Проверить базовую согласованность результата выбора цели."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0.0, 1.0].")
