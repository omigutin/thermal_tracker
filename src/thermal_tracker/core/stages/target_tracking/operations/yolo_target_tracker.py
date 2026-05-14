from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, Self

from ..result import TargetTrackingResult
from ....config import NeuralConfig, PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame, TrackerState
from ....nnet_interface import YoloNnetInterface
from ...frame_stabilization import FrameStabilizerResult
from .base_target_tracker import BaseTargetTracker
from ..type import TargetTrackerType


class YoloDetection(Protocol):
    """Описывает минимальные данные нейросетевой детекции для YOLO-трекинга."""

    bbox: BoundingBox
    confidence: float
    track_id: int | None
    class_id: int | None


@dataclass(frozen=True, slots=True)
class YoloTargetTrackerConfig:
    """Хранит настройки трекера цели поверх YOLO.track()."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetTrackerType] = TargetTrackerType.YOLO_TRACK

    # Конфигурация нейросетевого YOLO-интерфейса.
    neural_config: NeuralConfig | None = None

    # Радиус поиска ближайшей детекции при стартовом клике.
    click_search_radius: int = 32
    # Максимальное количество кадров потери перед сбросом трека.
    max_lost_frames: int = 15
    # Базовый отступ search-region вокруг последнего bbox.
    search_margin: int = 24
    # Рост search-region за каждый потерянный кадр.
    lost_search_growth: int = 8
    # Множитель максимальной дистанции для повторного захвата цели.
    max_reacquire_distance_factor: float = 2.5
    # Требовать совпадение класса при повторном захвате, если класс известен.
    prefer_same_class: bool = True

    # Вес Intersection over Union при повторном захвате.
    reacquire_iou_weight: float = 0.35
    # Вес близости к последнему bbox при повторном захвате.
    reacquire_distance_weight: float = 0.25
    # Вес похожести размера bbox при повторном захвате.
    reacquire_size_weight: float = 0.15

    def __post_init__(self) -> None:
        """Проверить корректность параметров YOLO-трекинга."""
        self._validate_positive_int(self.click_search_radius, "click_search_radius")
        self._validate_non_negative_int(self.max_lost_frames, "max_lost_frames")
        self._validate_non_negative_int(self.search_margin, "search_margin")
        self._validate_non_negative_int(self.lost_search_growth, "lost_search_growth")
        self._validate_positive_float(
            self.max_reacquire_distance_factor,
            "max_reacquire_distance_factor",
        )
        self._validate_non_negative_float(
            self.reacquire_iou_weight,
            "reacquire_iou_weight",
        )
        self._validate_non_negative_float(
            self.reacquire_distance_weight,
            "reacquire_distance_weight",
        )
        self._validate_non_negative_float(
            self.reacquire_size_weight,
            "reacquire_size_weight",
        )

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_bool_to(kwargs, "prefer_same_class")

        for field_name in (
            "click_search_radius",
            "max_lost_frames",
            "search_margin",
            "lost_search_growth",
        ):
            reader.pop_int_to(kwargs, field_name)

        for field_name in (
            "max_reacquire_distance_factor",
            "reacquire_iou_weight",
            "reacquire_distance_weight",
            "reacquire_size_weight",
        ):
            reader.pop_float_to(kwargs, field_name)

        reader.ensure_empty()
        return cls(**kwargs)

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

    @staticmethod
    def _validate_non_negative_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")

    @staticmethod
    def _validate_positive_float(value: float, field_name: str) -> None:
        """Проверить, что вещественное значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

    @staticmethod
    def _validate_non_negative_float(value: float, field_name: str) -> None:
        """Проверить, что вещественное значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")


@dataclass(slots=True)
class YoloTargetTracker(BaseTargetTracker):
    """Сопровождает одну выбранную цель поверх YOLO.track()."""

    config: YoloTargetTrackerConfig
    _engine: YoloNnetInterface = field(init=False, repr=False)

    _manual_track_id: int | None = field(default=None, init=False)
    _next_manual_track_id: int = field(default=0, init=False)
    _engine_track_id: int | None = field(default=None, init=False)
    _target_class_id: int | None = field(default=None, init=False)

    _bbox: BoundingBox | None = field(default=None, init=False)
    _predicted_bbox: BoundingBox | None = field(default=None, init=False)
    _search_region: BoundingBox | None = field(default=None, init=False)
    _score: float = field(default=0.0, init=False)
    _lost_frames: int = field(default=0, init=False)
    _message: str = field(default="Click detected target.", init=False)
    _state: TrackerState = field(default=TrackerState.IDLE, init=False)
    _latest_detections: tuple[YoloDetection, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Создать YOLO-интерфейс для нейросетевого трекинга."""
        if self.config.neural_config is None:
            raise ValueError("neural_config must be provided for YoloTargetTracker.")

        self._engine = YoloNnetInterface(self.config.neural_config)

    @property
    def latest_detections(self) -> tuple[YoloDetection, ...]:
        """Вернуть последние нейросетевые детекции для GUI и отладки."""
        return self._latest_detections

    def snapshot(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Вернуть текущее состояние трекера."""
        return TargetTrackingResult(
            state=self._state,
            track_id=self._manual_track_id,
            bbox=self._bbox,
            predicted_bbox=self._predicted_bbox,
            search_region=self._search_region,
            score=self._score,
            lost_frames=self._lost_frames,
            global_motion=motion,
            message=self._message,
        )

    def start_tracking(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
    ) -> TargetTrackingResult:
        """Начать сопровождение цели по клику по одной из YOLO-детекций."""
        if not self._latest_detections:
            self._latest_detections = tuple(self._engine.track(frame.bgr))

        candidate = self._select_detection_from_click(point)

        if candidate is None:
            self._state = TrackerState.IDLE
            self._message = "YOLO did not find a target near the click."
            self._score = 0.0
            return self.snapshot(FrameStabilizerResult())

        self._manual_track_id = self._next_manual_track_id
        self._next_manual_track_id += 1
        self._apply_candidate(candidate)
        self._message = (
            f"YOLO tracking target #{self._manual_track_id} "
            f"in {self._engine.mode_name} mode."
        )

        return self.snapshot(FrameStabilizerResult())

    def update(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
    ) -> TargetTrackingResult:
        """Обновить состояние YOLO-трекера по новому кадру."""
        self._latest_detections = tuple(self._engine.track(frame.bgr))

        if self._manual_track_id is None or self._bbox is None:
            self._state = TrackerState.IDLE
            self._score = 0.0
            self._search_region = None
            self._predicted_bbox = None
            self._message = self._build_idle_message()
            return self.snapshot(motion)

        exact_candidate = self._find_candidate_by_engine_track_id()

        if exact_candidate is not None:
            self._apply_candidate(exact_candidate)
            self._message = (
                f"YOLO tracks target #{self._manual_track_id} "
                f"in {self._engine.mode_name} mode."
            )
            return self.snapshot(motion)

        reacquired_candidate = self._find_reacquire_candidate()

        if reacquired_candidate is not None:
            self._apply_candidate(reacquired_candidate)
            self._message = f"Target #{self._manual_track_id} was reacquired by YOLO."
            return self.snapshot(motion)

        self._lost_frames += 1
        self._state = TrackerState.SEARCHING
        self._score = 0.0
        self._predicted_bbox = self._bbox
        self._search_region = self._expand_search_region(frame.bgr.shape)
        self._message = f"YOLO temporarily lost target #{self._manual_track_id}."

        if self._lost_frames > self.config.max_lost_frames:
            return self.reset()

        return self.snapshot(motion)

    def reset(self) -> TargetTrackingResult:
        """Сбросить текущее состояние YOLO-трекера."""
        manual_track_id = self._manual_track_id

        self._manual_track_id = None
        self._engine_track_id = None
        self._target_class_id = None
        self._bbox = None
        self._predicted_bbox = None
        self._search_region = None
        self._score = 0.0
        self._lost_frames = 0
        self._state = TrackerState.IDLE
        self._message = (
            f"Target #{manual_track_id} was reset. Click a new detected target."
            if manual_track_id is not None
            else "Click detected target."
        )

        return self.snapshot(FrameStabilizerResult())

    def resume_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        track_id: int,
    ) -> TargetTrackingResult:
        """Возобновить сопровождение цели с заданными bbox и track_id."""
        self._manual_track_id = track_id
        self._next_manual_track_id = max(self._next_manual_track_id, track_id + 1)
        self._engine_track_id = None
        self._target_class_id = None
        self._bbox = bbox.clamp(frame.bgr.shape)
        self._predicted_bbox = self._bbox
        self._search_region = self._bbox
        self._score = 1.0
        self._lost_frames = 0
        self._state = TrackerState.TRACKING
        self._message = f"YOLO resumed target #{self._manual_track_id}."

        return self.snapshot(FrameStabilizerResult())

    def _build_idle_message(self) -> str:
        """Вернуть сообщение для состояния ожидания выбора цели."""
        count = len(self._latest_detections)

        if count <= 0:
            return "No YOLO detections on the current frame."

        return f"Detected targets: {count}. Click the target to track."

    def _select_detection_from_click(
        self,
        point: tuple[int, int],
    ) -> YoloDetection | None:
        """Выбрать YOLO-детекцию по клику оператора."""
        containing = [
            detection
            for detection in self._latest_detections
            if self._point_inside_bbox(point=point, bbox=detection.bbox)
        ]

        if containing:
            return sorted(
                containing,
                key=lambda detection: (
                    detection.bbox.area,
                    -detection.confidence,
                ),
            )[0]

        best_candidate: YoloDetection | None = None
        best_distance = float("inf")

        for detection in self._latest_detections:
            center_x, center_y = detection.bbox.center
            distance = math.hypot(point[0] - center_x, point[1] - center_y)

            if (
                distance <= self.config.click_search_radius
                and distance < best_distance
            ):
                best_candidate = detection
                best_distance = distance

        return best_candidate

    def _find_candidate_by_engine_track_id(self) -> YoloDetection | None:
        """Найти детекцию с тем же track_id, который выдал внешний YOLO-трекер."""
        if self._engine_track_id is None:
            return None

        for detection in self._latest_detections:
            if detection.track_id == self._engine_track_id:
                return detection

        return None

    def _find_reacquire_candidate(self) -> YoloDetection | None:
        """Найти подходящую детекцию для повторного захвата цели."""
        if self._bbox is None:
            return None

        previous_bbox = self._bbox
        previous_center_x, previous_center_y = previous_bbox.center
        max_distance = (
            max(previous_bbox.width, previous_bbox.height)
            * self.config.max_reacquire_distance_factor
            + self._lost_frames * self.config.lost_search_growth
        )

        best_candidate: YoloDetection | None = None
        best_score = float("-inf")

        for detection in self._latest_detections:
            if not self._is_class_compatible(detection):
                continue

            candidate_center_x, candidate_center_y = detection.bbox.center
            distance = math.hypot(
                candidate_center_x - previous_center_x,
                candidate_center_y - previous_center_y,
            )

            if distance > max_distance:
                continue

            score = self._score_reacquire_candidate(
                detection=detection,
                previous_bbox=previous_bbox,
                distance=distance,
                max_distance=max_distance,
            )

            if score > best_score:
                best_score = score
                best_candidate = detection

        return best_candidate

    def _score_reacquire_candidate(
        self,
        detection: YoloDetection,
        previous_bbox: BoundingBox,
        distance: float,
        max_distance: float,
    ) -> float:
        """Рассчитать score кандидата для повторного захвата."""
        iou = detection.bbox.intersection_over_union(previous_bbox)
        distance_score = max(0.0, 1.0 - distance / max(max_distance, 1.0))
        size_ratio = min(detection.bbox.area, previous_bbox.area) / max(
            detection.bbox.area,
            previous_bbox.area,
            1,
        )

        return (
            detection.confidence
            + self.config.reacquire_iou_weight * iou
            + self.config.reacquire_distance_weight * distance_score
            + self.config.reacquire_size_weight * size_ratio
        )

    def _is_class_compatible(self, detection: YoloDetection) -> bool:
        """Проверить, подходит ли класс детекции для повторного захвата."""
        if not self.config.prefer_same_class:
            return True

        if self._target_class_id is None:
            return True

        if detection.class_id is None:
            return True

        return detection.class_id == self._target_class_id

    def _apply_candidate(self, candidate: YoloDetection) -> None:
        """Принять YOLO-детекцию как текущее положение цели."""
        self._bbox = candidate.bbox
        self._predicted_bbox = candidate.bbox
        self._search_region = candidate.bbox
        self._engine_track_id = candidate.track_id
        self._target_class_id = candidate.class_id
        self._score = candidate.confidence
        self._lost_frames = 0
        self._state = TrackerState.TRACKING

    def _expand_search_region(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox | None:
        """Расширить область поиска при временной потере цели."""
        if self._bbox is None:
            return None

        padding = (
            self.config.search_margin
            + self._lost_frames * self.config.lost_search_growth
        )

        return self._bbox.pad(padding, padding).clamp(frame_shape)

    @staticmethod
    def _point_inside_bbox(point: tuple[int, int], bbox: BoundingBox) -> bool:
        """Проверить, находится ли точка внутри bbox."""
        x, y = point
        return bbox.x <= x < bbox.x2 and bbox.y <= y < bbox.y2
