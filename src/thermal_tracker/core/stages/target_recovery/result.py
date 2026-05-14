from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import BoundingBox


@dataclass(slots=True)
class TargetRecoveryResult:
    """Результат стадии повторного захвата потерянной цели."""

    bbox: BoundingBox | None
    score: float = 0.0
    search_region: BoundingBox | None = None
    source_name: str = ""
    message: str = ""

    @property
    def recovered(self) -> bool:
        """Проверить, была ли цель повторно найдена."""
        return self.bbox is not None
