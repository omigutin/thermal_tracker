"""Диагностика детектора кандидатов цели."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CandidateInfo:
    """Кандидат цели после выделения на кадре."""
    center_x: float
    center_y: float
    area: float
    width: float
    height: float
    score: float | None = None
    dist_from_reference: float | None = None

@dataclass(frozen=True)
class DetectionDiagnostics:
    """Результаты работы детектора и выбора кандидата."""
    threshold: float | None = None
    candidates: list[CandidateInfo] = field(default_factory=list)
    picked_index: int | None = None
    contrast_stats_around: dict[str, float] | None = None
