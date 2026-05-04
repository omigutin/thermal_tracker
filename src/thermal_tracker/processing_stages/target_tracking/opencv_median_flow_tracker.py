"""Заготовка под OpenCV MedianFlow tracker."""

from __future__ import annotations

from .base_target_tracker import BaseSingleTargetTracker


class MedianFlowSingleTargetTracker(BaseSingleTargetTracker):
    """Будущий baseline для мягких сценариев без тяжёлых окклюзий."""

    implementation_name = "median_flow"
    is_ready = False

    def snapshot(self, motion):
        raise NotImplementedError("MedianFlow tracker пока не реализован.")

    def start_tracking(self, frame, point):
        raise NotImplementedError("MedianFlow tracker пока не реализован.")

    def update(self, frame, motion):
        raise NotImplementedError("MedianFlow tracker пока не реализован.")

    def reset(self):
        raise NotImplementedError("MedianFlow tracker пока не реализован.")
