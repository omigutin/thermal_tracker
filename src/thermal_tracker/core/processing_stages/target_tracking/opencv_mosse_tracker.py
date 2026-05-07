"""Заготовка под OpenCV MOSSE tracker."""

from __future__ import annotations

from .base_target_tracker import BaseSingleTargetTracker


class MosseSingleTargetTracker(BaseSingleTargetTracker):
    """Будущий очень быстрый baseline для дешёвых платформ."""

    def snapshot(self, motion):
        raise NotImplementedError("MOSSE tracker пока не реализован.")

    def start_tracking(self, frame, point):
        raise NotImplementedError("MOSSE tracker пока не реализован.")

    def update(self, frame, motion):
        raise NotImplementedError("MOSSE tracker пока не реализован.")

    def reset(self):
        raise NotImplementedError("MOSSE tracker пока не реализован.")
