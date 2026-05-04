"""Заготовка под фильтр по согласованности движения."""

from __future__ import annotations

from .base_candidate_filter import BaseTargetFilter


class MotionConsistencyTargetFilter(BaseTargetFilter):
    """Будущий фильтр для отбора объектов с более правдоподобным motion-поведением."""

    implementation_name = "motion_consistency"
    is_ready = False

    def filter(self, frame, objects, motion):
        raise NotImplementedError("Motion-consistency filter пока не реализован.")
