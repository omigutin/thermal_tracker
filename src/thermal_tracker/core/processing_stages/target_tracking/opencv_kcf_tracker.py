"""Заготовка под OpenCV KCF tracker."""

from __future__ import annotations

from .base_target_tracker import BaseSingleTargetTracker


class KcfSingleTargetTracker(BaseSingleTargetTracker):
    """Будущий KCF-baseline для быстрых сравнений."""

    def snapshot(self, motion):
        raise NotImplementedError("KCF tracker пока не реализован.")

    def start_tracking(self, frame, point):
        raise NotImplementedError("KCF tracker пока не реализован.")

    def update(self, frame, motion):
        raise NotImplementedError("KCF tracker пока не реализован.")

    def reset(self):
        raise NotImplementedError("KCF tracker пока не реализован.")
