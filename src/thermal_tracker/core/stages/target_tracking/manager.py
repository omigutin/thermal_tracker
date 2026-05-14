"""Менеджер выбора single-target трекера."""

from __future__ import annotations

from ...config import ClickSelectionConfig, IrstTrackerConfig, NeuralConfig, OpenCVTrackerConfig, YoloTrackerConfig
from ...domain.models import BoundingBox, ProcessedFrame, TrackSnapshot
from ..candidate_formation.result import DetectedObject
from ..frame_stabilization.result import FrameStabilizerResult
from thermal_tracker.core.stages.target_tracking.operations.base_target_tracker import BaseSingleTargetTracker
from thermal_tracker.core.stages.target_tracking.operations.irst_contrast_target_tracker import IrstSingleTargetTracker
from thermal_tracker.core.stages.target_tracking.operations.yolo_target_tracker import YoloTrackSingleTargetTracker
from thermal_tracker.core.stages.target_tracking.operations.template_point_target_tracker import ClickToTrackSingleTargetTracker
from .type import TargetTrackerType


TargetTrackerInput = TargetTrackerType | str


class TargetTrackerManager:
    """Создаёт и запускает выбранный трекер одной цели."""

    def __init__(
        self,
        tracker: TargetTrackerInput,
        tracker_config: OpenCVTrackerConfig | YoloTrackerConfig | IrstTrackerConfig,
        click_config: ClickSelectionConfig,
        neural_config: NeuralConfig | None = None,
    ) -> None:
        self._tracker = self._build_tracker(tracker, tracker_config, click_config, neural_config)

    @property
    def tracker(self) -> BaseSingleTargetTracker:
        """Возвращает подготовленный трекер."""

        return self._tracker

    def snapshot(self, motion: FrameStabilizerResult) -> TrackSnapshot:
        """Возвращает состояние выбранного трекера."""

        return self._tracker.snapshot(motion)

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает сопровождение выбранной цели."""

        return self._tracker.start_tracking(frame, point)

    def update(self, frame: ProcessedFrame, motion: FrameStabilizerResult) -> TrackSnapshot:
        """Обновляет состояние трекера на новом кадре."""

        return self._tracker.update(frame, motion)

    def reset(self) -> TrackSnapshot:
        """Сбрасывает состояние трекера."""

        return self._tracker.reset()

    def resume_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        track_id: int,
    ) -> TrackSnapshot:
        """Возобновляет сопровождение цели с тем же track_id после recovery."""

        return self._tracker.resume_tracking(frame, bbox, track_id)

    @property
    def latest_detections(self) -> tuple[DetectedObject, ...]:
        """Возвращает последний набор нейросетевых детекций.

        Не пусто только у YoloTrackSingleTargetTracker. Для OpenCV-трекера
        возвращает пустой кортеж, чтобы вызывающий код мог не делать isinstance.
        """

        if isinstance(self._tracker, YoloTrackSingleTargetTracker):
            return self._tracker.latest_detections
        return ()

    @classmethod
    def _build_tracker(
        cls,
        tracker: TargetTrackerInput,
        tracker_config: OpenCVTrackerConfig | YoloTrackerConfig | IrstTrackerConfig,
        click_config: ClickSelectionConfig,
        neural_config: NeuralConfig | None,
    ) -> BaseSingleTargetTracker:
        tracker_type = cls._normalize_tracker_type(tracker)
        if tracker_type == TargetTrackerType.OPENCV_TEMPLATE_POINT:
            if not isinstance(tracker_config, OpenCVTrackerConfig):
                raise TypeError(
                    f"OpenCV template tracker requires OpenCVTrackerConfig, "
                    f"got {type(tracker_config).__name__}."
                )
            return ClickToTrackSingleTargetTracker(tracker_config, click_config)
        if tracker_type == TargetTrackerType.NN_YOLO:
            if not isinstance(tracker_config, YoloTrackerConfig):
                raise TypeError(
                    f"YOLO tracker requires YoloTrackerConfig, "
                    f"got {type(tracker_config).__name__}."
                )
            if neural_config is None:
                raise ValueError("NN YOLO tracker requires neural config.")
            return YoloTrackSingleTargetTracker(tracker_config, click_config, neural_config)
        if tracker_type == TargetTrackerType.IRST_CONTRAST:
            if not isinstance(tracker_config, IrstTrackerConfig):
                raise TypeError(
                    f"IRST tracker requires IrstTrackerConfig, "
                    f"got {type(tracker_config).__name__}."
                )
            return IrstSingleTargetTracker(tracker_config, click_config)
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
