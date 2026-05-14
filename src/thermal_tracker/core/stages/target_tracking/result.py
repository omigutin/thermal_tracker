from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import BoundingBox, TrackerState
from ..frame_stabilization import FrameStabilizerResult


@dataclass(slots=True)
class TargetTrackingResult:
    """Результат стадии сопровождения выбранной цели."""

    state: TrackerState
    track_id: int | None
    bbox: BoundingBox | None
    predicted_bbox: BoundingBox | None
    search_region: BoundingBox | None
    score: float
    lost_frames: int
    global_motion: FrameStabilizerResult
    message: str = ""
