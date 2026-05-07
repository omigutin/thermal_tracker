"""Результат поиска цели на текущем кадре."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import BoundingBox


@dataclass
class SearchResult:
    """Лучший найденный кандидат на текущем кадре."""

    bbox: BoundingBox
    score: float
    search_region: BoundingBox
