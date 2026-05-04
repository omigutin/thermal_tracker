"""Заготовка под чистый template tracker."""

from __future__ import annotations

from .base_target_tracker import BaseSingleTargetTracker


class TemplateSingleTargetTracker(BaseSingleTargetTracker):
    """Будущий трекер, который опирается почти только на внешний вид цели."""

    implementation_name = "template_only"
    is_ready = False

    def snapshot(self, motion):
        raise NotImplementedError("Template-only tracker пока не реализован.")

    def start_tracking(self, frame, point):
        raise NotImplementedError("Template-only tracker пока не реализован.")

    def update(self, frame, motion):
        raise NotImplementedError("Template-only tracker пока не реализован.")

    def reset(self):
        raise NotImplementedError("Template-only tracker пока не реализован.")
