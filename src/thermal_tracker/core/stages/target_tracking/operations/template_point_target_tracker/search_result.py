from __future__ import annotations

from dataclasses import dataclass

from .....domain.models import BoundingBox


@dataclass(slots=True)
class SearchResult:
    """Лучший найденный кандидат цели внутри template-point трекера."""

    bbox: BoundingBox
    score: float
    search_region: BoundingBox
