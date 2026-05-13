"""Заготовка под multiscale-повторный захват."""

from __future__ import annotations

from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame
from .base_target_recoverer import BaseReacquirer


class MultiScaleReacquirer(BaseReacquirer):
    """Будущий reacquirer с перебором нескольких масштабов и областей поиска."""

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
        lost_frames: int = 0,
    ) -> BoundingBox | None:
        raise NotImplementedError("Multi-scale reacquirer is not implemented yet.")
