"""Прогноз положения цели по опорным точкам."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import BoundingBox


@dataclass
class PointPrediction:
    """Прогноз положения цели по опорным точкам."""

    bbox: BoundingBox
    confidence: float
