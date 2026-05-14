from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import BoundingBox, ProcessedFrame, TrackerState
from ..frame_stabilization import FrameStabilizerResult
from .config import TargetTrackerConfig
from .factory import TargetTrackerFactory
from .operations import BaseTargetTracker
from .result import TargetTrackingResult


class TargetTrackingManager:
    """Управляет выполнением выбранного трекера цели."""

    def __init__(self, operations: Sequence[TargetTrackerConfig]) -> None:
        """Создать менеджер и подготовить активные runtime-трекеры."""
        self._trackers: tuple[BaseTargetTracker, ...] = (
            TargetTrackerFactory.build_many(operations)
        )
        self._validate_trackers_count()

    @property
    def trackers(self) -> tuple[BaseTargetTracker, ...]:
        """Вернуть подготовленные runtime-трекеры."""
        return self._trackers

    @property
    def tracker(self) -> BaseTargetTracker | None:
        """Вернуть активный трекер, если он есть."""
        if not self._trackers:
            return None

        return self._trackers[0]

    def snapshot(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Вернуть текущее состояние трекера."""
        if self.tracker is None:
            return self._build_idle_result(
                motion=motion,
                message="Target tracker is disabled.",
            )

        return self.tracker.snapshot(motion)

    def start_tracking(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
    ) -> TargetTrackingResult:
        """Начать сопровождение цели по точке выбора."""
        if self.tracker is None:
            return self._build_idle_result(
                motion=FrameStabilizerResult(),
                message="Target tracker is disabled.",
            )

        return self.tracker.start_tracking(
            frame=frame,
            point=point,
        )

    def update(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
    ) -> TargetTrackingResult:
        """Обновить состояние активного трекера по новому кадру."""
        if self.tracker is None:
            return self._build_idle_result(
                motion=motion,
                message="Target tracker is disabled.",
            )

        return self.tracker.update(
            frame=frame,
            motion=motion,
        )

    def reset(self) -> TargetTrackingResult:
        """Сбросить активный трекер."""
        if self.tracker is None:
            return self._build_idle_result(
                motion=FrameStabilizerResult(),
                message="Target tracker is disabled.",
            )

        return self.tracker.reset()

    def resume_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        track_id: int,
    ) -> TargetTrackingResult:
        """Возобновить сопровождение цели с заданными bbox и track_id."""
        if self.tracker is None:
            return self._build_idle_result(
                motion=FrameStabilizerResult(),
                message="Target tracker is disabled.",
            )

        return self.tracker.resume_tracking(
            frame=frame,
            bbox=bbox,
            track_id=track_id,
        )

    def _validate_trackers_count(self) -> None:
        """Проверить, что активен не более одного трекера цели."""
        if len(self._trackers) > 1:
            raise ValueError(
                "Target tracking supports only one active tracker. "
                "Tracker fallback chain is not implemented."
            )

    @staticmethod
    def _build_idle_result(
        motion: FrameStabilizerResult,
        message: str,
    ) -> TargetTrackingResult:
        """Создать пустой результат для выключенного трекера."""
        return TargetTrackingResult(
            state=TrackerState.IDLE,
            track_id=None,
            bbox=None,
            predicted_bbox=None,
            search_region=None,
            score=0.0,
            lost_frames=0,
            global_motion=motion,
            message=message,
        )
