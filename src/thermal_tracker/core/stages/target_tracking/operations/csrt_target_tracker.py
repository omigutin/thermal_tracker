from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, Self

import cv2
import numpy as np

from ....config.preset_field_reader import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame, TrackerState
from ..result import TargetTrackingResult
from ...frame_stabilization import FrameStabilizerResult
from ...target_selection.config import TargetSelectionConfig
from ...target_selection.factory import TargetSelectionFactory
from ...target_selection.operations import BaseTargetSelector, ContrastComponentTargetSelectorConfig
from .base_target_tracker import BaseTargetTracker
from ..type import TargetTrackerType


class _OpenCvTracker(Protocol):
    """Описывает минимальный интерфейс OpenCV-трекера."""

    def init(self, image: np.ndarray, bounding_box: tuple[int, int, int, int]) -> bool:
        """Инициализировать OpenCV-трекер."""
        ...

    def update(self, image: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        """Обновить OpenCV-трекер по новому кадру."""
        ...


class _OpenCvTrackerFactory(Protocol):
    """Описывает фабрику OpenCV-трекера."""

    def __call__(self) -> _OpenCvTracker:
        """Создать OpenCV-трекер."""
        ...


@dataclass(frozen=True, slots=True)
class CsrtTargetTrackerConfig:
    """Хранит настройки CSRT-трекера одной цели."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetTrackerType] = TargetTrackerType.CSRT
    # Максимальное количество кадров, которое трекер может быть в состоянии потери.
    max_lost_frames: int = 10
    # Конфигурация выбора цели по первому клику.
    target_selector_config: TargetSelectionConfig = field(default_factory=ContrastComponentTargetSelectorConfig)

    def __post_init__(self) -> None:
        """Проверить корректность параметров CSRT-трекера."""
        if self.max_lost_frames < 0:
            raise ValueError("max_lost_frames must be greater than or equal to 0.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "max_lost_frames")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class CsrtTargetTracker(BaseTargetTracker):
    """Сопровождает одну цель через OpenCV CSRT-трекер."""

    config: CsrtTargetTrackerConfig
    _target_selector: BaseTargetSelector = field(init=False, repr=False)
    _tracker_factory: _OpenCvTrackerFactory | None = field(init=False, repr=False)
    _tracker: _OpenCvTracker | None = field(default=None, init=False, repr=False)

    _track_id: int | None = field(default=None, init=False)
    _next_track_id: int = field(default=0, init=False)
    _bbox: BoundingBox | None = field(default=None, init=False)
    _predicted_bbox: BoundingBox | None = field(default=None, init=False)
    _search_region: BoundingBox | None = field(default=None, init=False)
    _lost_frames: int = field(default=0, init=False)
    _score: float = field(default=0.0, init=False)
    _state: TrackerState = field(default=TrackerState.IDLE, init=False)
    _message: str = field(default="Click target", init=False)

    def __post_init__(self) -> None:
        """Подготовить OpenCV CSRT и selector для выбора цели."""
        target_selector = TargetSelectionFactory.build(self.config.target_selector_config)

        if target_selector is None:
            raise ValueError("Target selector config is disabled.")

        self._target_selector = target_selector
        self._tracker_factory = self._resolve_csrt_factory()

    def snapshot(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Вернуть текущее состояние трекера."""
        return TargetTrackingResult(
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

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TargetTrackingResult:
        """Начать сопровождение цели по точке выбора."""
        if self._tracker_factory is None:
            raise RuntimeError("OpenCV CSRT tracker is not available in this OpenCV build.")

        selection = self._target_selector.apply(frame=frame, point=point)
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

        return self.snapshot(FrameStabilizerResult())

    def update(self, frame: ProcessedFrame, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Обновить состояние CSRT-трекера по новому кадру."""
        if self._tracker is None or self._bbox is None:
            self._state = TrackerState.IDLE
            self._message = "Click target"
            return self.snapshot(motion)

        ok, raw_bbox = self._tracker.update(frame.bgr)

        if ok:
            candidate = self._build_bbox_from_raw(raw_bbox, frame.bgr.shape)
            refined = self._try_refine(frame=frame, bbox=candidate)

            self._bbox = refined if refined is not None else candidate
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

    def reset(self) -> TargetTrackingResult:
        """Сбросить текущее состояние CSRT-трекера."""
        self._tracker = None
        self._track_id = None
        self._bbox = None
        self._predicted_bbox = None
        self._search_region = None
        self._lost_frames = 0
        self._score = 0.0
        self._state = TrackerState.IDLE
        self._message = "Tracker reset"

        return self.snapshot(FrameStabilizerResult())

    def resume_tracking(self, frame: ProcessedFrame, bbox: BoundingBox, track_id: int) -> TargetTrackingResult:
        """Возобновить сопровождение цели с заданными bbox и track_id."""
        if self._tracker_factory is None:
            raise RuntimeError("OpenCV CSRT tracker is not available in this OpenCV build.")

        clamped_bbox = bbox.clamp(frame.bgr.shape)
        tracker = self._tracker_factory()
        tracker.init(frame.bgr, clamped_bbox.to_xywh())

        self._tracker = tracker
        self._track_id = track_id
        self._next_track_id = max(self._next_track_id, track_id + 1)
        self._bbox = clamped_bbox
        self._predicted_bbox = clamped_bbox
        self._search_region = clamped_bbox
        self._lost_frames = 0
        self._score = 1.0
        self._state = TrackerState.TRACKING
        self._message = f"Resumed target #{self._track_id} with CSRT"

        return self.snapshot(FrameStabilizerResult())

    def _try_refine(self, frame: ProcessedFrame, bbox: BoundingBox) -> BoundingBox | None:
        """Попробовать уточнить bbox через selector, если он поддерживает refine."""
        try:
            refined = self._target_selector.refine(frame=frame, bbox=bbox)
        except NotImplementedError:
            return None

        if refined is None:
            return None

        return refined.bbox.clamp(frame.bgr.shape)

    @staticmethod
    def _build_bbox_from_raw(
        raw_bbox: tuple[float, float, float, float],
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Преобразовать bbox OpenCV в доменную модель."""
        x, y, width, height = raw_bbox

        return BoundingBox(
            x=int(round(x)),
            y=int(round(y)),
            width=max(1, int(round(width))),
            height=max(1, int(round(height))),
        ).clamp(frame_shape)

    @staticmethod
    def _resolve_csrt_factory() -> _OpenCvTrackerFactory | None:
        """Найти фабрику CSRT-трекера в текущей сборке OpenCV."""
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create

        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create

        return None
