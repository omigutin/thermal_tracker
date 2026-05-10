"""Заготовка под почти глобальный повторный захват."""

from __future__ import annotations

from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame
from .base_target_recoverer import BaseReacquirer


class GlobalReacquirer(BaseReacquirer):
    """Будущий fallback-режим, когда локальный поиск уже не помог."""

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
        lost_frames: int = 0,
    ) -> BoundingBox | None:
        raise NotImplementedError("Global reacquirer is not implemented yet.")
