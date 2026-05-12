"""Обёртка над OpenCV CSRT-трекером."""

from __future__ import annotations

import cv2

from ...config import ClickSelectionConfig, OpenCVTrackerConfig
from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame, TrackSnapshot, TrackerState
from ..target_selection import ClickTargetSelector
from .base_target_tracker import BaseSingleTargetTracker


def _resolve_csrt_factory():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create
    return None


class CsrtSingleTargetTracker(BaseSingleTargetTracker):
    """Трекер одной цели на базе готового CSRT."""

    def __init__(self, tracker_config: OpenCVTrackerConfig, click_config: ClickSelectionConfig) -> None:
        self.config = tracker_config
        self.selector = ClickTargetSelector(click_config)
        self._tracker_factory = _resolve_csrt_factory()
        self._tracker = None
        self._track_id: int | None = None
        self._next_track_id = 0
        self._bbox: BoundingBox | None = None
        self._predicted_bbox: BoundingBox | None = None
        self._search_region: BoundingBox | None = None
        self._lost_frames = 0
        self._score = 0.0
        self._state = TrackerState.IDLE
        self._message = "Click target"

    def snapshot(self, motion: GlobalMotion) -> TrackSnapshot:
        return TrackSnapshot(
            state=self._state,
            track_id=self._track_id,
            bbox=self._bbox,
            predicted_bbox=self._predicted_bbox,
            search_region=self._search_region,
            score=self._score,
            lost_frames=self._lost_frames,
            global_motion=motion,
            message=self._message,
        )

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        if self._tracker_factory is None:
            raise RuntimeError("В этой сборке OpenCV нет CSRT tracker.")

        selection = self.selector.select(frame, point)
        bbox = selection.bbox.clamp(frame.bgr.shape)
        tracker = self._tracker_factory()
        tracker.init(frame.bgr, bbox.to_xywh())

        self._tracker = tracker
        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._bbox = bbox
        self._predicted_bbox = bbox
        self._search_region = bbox
        self._lost_frames = 0
        self._score = 1.0
        self._state = TrackerState.TRACKING
        self._message = f"Tracking target #{self._track_id} with CSRT"
        return self.snapshot(GlobalMotion())

    def update(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        if self._tracker is None or self._bbox is None:
            self._state = TrackerState.IDLE
            self._message = "Click target"
            return self.snapshot(motion)

        ok, raw_bbox = self._tracker.update(frame.bgr)
        if ok:
            x, y, w, h = raw_bbox
            candidate = BoundingBox(
                x=int(round(x)),
                y=int(round(y)),
                width=max(1, int(round(w))),
                height=max(1, int(round(h))),
            ).clamp(frame.bgr.shape)
            refined = self.selector.refine(frame, candidate)
            self._bbox = refined.bbox if refined is not None else candidate
            self._predicted_bbox = self._bbox
            self._search_region = self._bbox
            self._lost_frames = 0
            self._score = 1.0
            self._state = TrackerState.TRACKING
            self._message = f"Tracking target #{self._track_id} with CSRT"
            return self.snapshot(motion)

        self._lost_frames += 1
        self._state = TrackerState.SEARCHING
        self._score = 0.0
        self._message = f"CSRT lost target #{self._track_id}"
        if self._lost_frames > self.config.max_lost_frames:
            return self.reset()
        return self.snapshot(motion)

    def reset(self) -> TrackSnapshot:
        self._tracker = None
        self._track_id = None
        self._bbox = None
        self._predicted_bbox = None
        self._search_region = None
        self._lost_frames = 0
        self._score = 0.0
        self._state = TrackerState.IDLE
        self._message = "Tracker reset"
        return self.snapshot(GlobalMotion())
