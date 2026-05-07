"""Заготовка под трекер только по опорным точкам."""

from __future__ import annotations

from .base_target_tracker import BaseSingleTargetTracker


class PointFlowSingleTargetTracker(BaseSingleTargetTracker):
    """Будущий трекер, который опирается в основном на optical flow по точкам."""

    implementation_name = "point_flow"
    is_ready = False

    def snapshot(self, motion):
        raise NotImplementedError("Point-flow tracker пока не реализован.")

    def start_tracking(self, frame, point):
        raise NotImplementedError("Point-flow tracker пока не реализован.")

    def update(self, frame, motion):
        raise NotImplementedError("Point-flow tracker пока не реализован.")

    def reset(self):
        raise NotImplementedError("Point-flow tracker пока не реализован.")
