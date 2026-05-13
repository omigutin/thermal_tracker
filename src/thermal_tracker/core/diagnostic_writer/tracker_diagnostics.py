"""Диагностика шага обновления трекера."""
from __future__ import annotations
from dataclasses import dataclass, field
from .detection_diagnostics import CandidateInfo

@dataclass(frozen=True)
class TrackerUpdateDiagnostics:
    """Подробности выбора кандидата на шаге трекинга."""
    predicted_center: tuple[float, float] | None = None
    gate_radius: float | None = None
    max_physical_dist: float | None = None
    candidates: list[CandidateInfo] = field(default_factory=list)
    picked_index: int | None = None
    lost_frames: int | None = None
