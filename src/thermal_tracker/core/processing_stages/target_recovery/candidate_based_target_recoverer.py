"""Заготовка под повторный захват через список кандидатов."""

from __future__ import annotations

from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame
from .base_target_recoverer import BaseReacquirer


class CandidateBasedReacquirer(BaseReacquirer):
    """Будущий reacquirer, который работает поверх списка объектов-кандидатов."""

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
    ) -> BoundingBox | None:
        raise NotImplementedError("Candidate-based reacquirer пока не реализован.")
