from __future__ import annotations

from dataclasses import dataclass

from .....domain.models import BoundingBox


@dataclass(slots=True)
class PointPrediction:
    """Прогноз положения цели по опорным точкам."""

    bbox: BoundingBox
    confidence: float

    def __post_init__(self) -> None:
        """Проверить корректность прогноза по точкам."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0.0, 1.0].")
