"""Менеджер выбора single-target трекера."""

from __future__ import annotations

from ...config import ClickSelectionConfig, NeuralConfig, TrackerConfig
from ...domain.models import GlobalMotion, ProcessedFrame, TrackSnapshot
from .base_target_tracker import BaseSingleTargetTracker
from .nn_yolo_target_tracker import YoloTrackSingleTargetTracker
from .opencv_template_point_target_tracker import ClickToTrackSingleTargetTracker
from .target_tracker_type import TargetTrackerType


TargetTrackerInput = TargetTrackerType | str


class TargetTrackerManager:
    """Создаёт и запускает выбранный трекер одной цели."""

    def __init__(
        self,
        tracker: TargetTrackerInput,
        tracker_config: TrackerConfig,
        click_config: ClickSelectionConfig,
        neural_config: NeuralConfig | None = None,
    ) -> None:
        self._tracker = self._build_tracker(tracker, tracker_config, click_config, neural_config)

    @property
    def tracker(self) -> BaseSingleTargetTracker:
        """Возвращает подготовленный трекер."""

        return self._tracker

    def snapshot(self, motion: GlobalMotion) -> TrackSnapshot:
        """Возвращает состояние выбранного трекера."""

        return self._tracker.snapshot(motion)

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает сопровождение выбранной цели."""

        return self._tracker.start_tracking(frame, point)

    def update(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        """Обновляет состояние трекера на новом кадре."""

        return self._tracker.update(frame, motion)

    def reset(self) -> TrackSnapshot:
        """Сбрасывает состояние трекера."""

        return self._tracker.reset()

    def __getattr__(self, name: str):
        return getattr(self._tracker, name)

    @classmethod
    def _build_tracker(
        cls,
        tracker: TargetTrackerInput,
        tracker_config: TrackerConfig,
        click_config: ClickSelectionConfig,
        neural_config: NeuralConfig | None,
    ) -> BaseSingleTargetTracker:
        tracker_type = cls._normalize_tracker_type(tracker)
        if tracker_type == TargetTrackerType.OPENCV_TEMPLATE_POINT:
            return ClickToTrackSingleTargetTracker(tracker_config, click_config)
        if tracker_type == TargetTrackerType.NN_YOLO:
            if neural_config is None:
                raise ValueError("NN YOLO tracker requires neural config.")
            return YoloTrackSingleTargetTracker(tracker_config, click_config, neural_config)
        raise ValueError(f"Unsupported target tracker type: {tracker_type!r}.")

    @staticmethod
    def _normalize_tracker_type(tracker: TargetTrackerInput) -> TargetTrackerType:
        if isinstance(tracker, TargetTrackerType):
            return tracker
        try:
            return TargetTrackerType(tracker)
        except ValueError:
            pass
        tracker_by_name = TargetTrackerType.__members__.get(tracker.upper())
        if tracker_by_name is not None:
            return tracker_by_name
        raise ValueError(
            f"Unsupported target tracker value: {tracker!r}. "
            f"Available values: {tuple(item.value for item in TargetTrackerType)}."
        )
