"""Диагностика этапа обработки клика пользователя."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ClickDiagnostics:
    """Снимок параметров клика и зоны поиска цели."""
    frame_x: int | None = None
    frame_y: int | None = None
    search_region: tuple[int, int, int, int] | None = None
    click_search_radius: int | None = None
    click_fallback_size: int | None = None
