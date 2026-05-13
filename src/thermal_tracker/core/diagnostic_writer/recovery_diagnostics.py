"""Диагностика этапа восстановления потерянной цели."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RecoveryDiagnostics:
    """Состояние recoverer во время восстановления трека."""
    state: str | None = None
    pending_age_frames: int | None = None
    pending_confirmations: int | None = None
    recoverer_kind: str | None = None
    candidate_found: bool | None = None
