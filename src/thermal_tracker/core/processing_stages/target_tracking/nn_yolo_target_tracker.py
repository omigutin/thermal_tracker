"""Трекер одной цели на базе YOLO.track() и внешнего многообъектного трекера.

Логика простая:
- каждый кадр прогоняем через NN-детектор + ByteTrack;
- пользователь кликом выбирает одну из найденных целей;
- дальше держимся за её track id;
- если id потерялся, пытаемся вернуть цель по классу, близости и размеру бокса.
"""

from __future__ import annotations

import math

from ...config import ClickSelectionConfig, NeuralConfig, TrackerConfig
from ...domain.models import BoundingBox, DetectedObject, GlobalMotion, ProcessedFrame, TrackSnapshot, TrackerState
from ...nnet_interface import YoloNnetInterface
from .base_target_tracker import BaseSingleTargetTracker


def _point_inside_bbox(point: tuple[int, int], bbox: BoundingBox) -> bool:
    x, y = point
    return bbox.x <= x < bbox.x2 and bbox.y <= y < bbox.y2


class YoloTrackSingleTargetTracker(BaseSingleTargetTracker):
    """Ведёт одну выбранную цель поверх общего NN-потока сопровождения."""

    implementation_name = "yolo_track_single_target"
    is_ready = True

    def __init__(
        self,
        tracker_config: TrackerConfig,
        click_config: ClickSelectionConfig,
        neural_config: NeuralConfig,
    ) -> None:
        self.config = tracker_config
        self.click_config = click_config
        self.neural_config = neural_config
        self.engine = YoloNnetInterface(neural_config)

        self._manual_track_id: int | None = None
        self._next_manual_track_id = 0
        self._engine_track_id: int | None = None
        self._target_class_id: int | None = None
        self._bbox: BoundingBox | None = None
        self._predicted_bbox: BoundingBox | None = None
        self._search_region: BoundingBox | None = None
        self._score = 0.0
        self._lost_frames = 0
        self._message = "Кликните по найденной цели."
        self._state: TrackerState = TrackerState.IDLE
        self._latest_detections: list[DetectedObject] = []

    @property
    def latest_detections(self) -> tuple[DetectedObject, ...]:
        """Возвращает текущий набор нейросетевых кандидатов для GUI и отладки."""

        return tuple(self._latest_detections)

    def snapshot(self, motion: GlobalMotion) -> TrackSnapshot:
        return TrackSnapshot(
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

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        if not self._latest_detections:
            self._latest_detections = self.engine.track(frame.bgr)

        candidate = self._select_detection_from_click(point)
        if candidate is None:
            self._state = TrackerState.IDLE
            self._message = "Нейросеть не нашла цель рядом с кликом."
            self._score = 0.0
            return self.snapshot(GlobalMotion())

        self._manual_track_id = self._next_manual_track_id
        self._next_manual_track_id += 1
        self._apply_candidate(candidate, reacquired=False)
        self._message = (
            f"Нейросетевой трекинг цели #{self._manual_track_id} запущен "
            f"в режиме {self.engine.mode_name}."
        )
        return self.snapshot(GlobalMotion())

    def update(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        self._latest_detections = self.engine.track(frame.bgr)

        if self._manual_track_id is None or self._bbox is None:
            self._state = TrackerState.IDLE
            self._score = 0.0
            self._search_region = None
            self._predicted_bbox = None
            self._message = self._build_idle_message()
            return self.snapshot(motion)

        exact_candidate = self._find_candidate_by_engine_track_id()
        if exact_candidate is not None:
            self._apply_candidate(exact_candidate, reacquired=False)
            self._message = (
                f"Нейросеть ведёт цель #{self._manual_track_id} "
                f"в режиме {self.engine.mode_name}."
            )
            return self.snapshot(motion)

        reacquired_candidate = self._find_reacquire_candidate()
        if reacquired_candidate is not None:
            self._apply_candidate(reacquired_candidate, reacquired=True)
            self._message = f"Цель #{self._manual_track_id} повторно захвачена нейросетью."
            return self.snapshot(motion)

        self._lost_frames += 1
        self._state = TrackerState.SEARCHING
        self._score = 0.0
        self._predicted_bbox = self._bbox
        self._search_region = self._expand_search_region(frame.bgr.shape)
        self._message = f"Нейросеть временно потеряла цель #{self._manual_track_id}."
        if self._lost_frames > self.config.max_lost_frames:
            return self.reset()
        return self.snapshot(motion)

    def reset(self) -> TrackSnapshot:
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
            f"Цель #{manual_track_id} сброшена. Кликните по новой цели."
            if manual_track_id is not None
            else "Кликните по найденной цели."
        )
        return self.snapshot(GlobalMotion())

    def _build_idle_message(self) -> str:
        count = len(self._latest_detections)
        if count <= 0:
            return "На кадре пока нет нейросетевых детекций."
        return f"Найдено целей: {count}. Кликните по нужной."

    def _select_detection_from_click(self, point: tuple[int, int]) -> DetectedObject | None:
        containing = [det for det in self._latest_detections if _point_inside_bbox(point, det.bbox)]
        if containing:
            containing.sort(key=lambda det: (det.bbox.area, -det.confidence))
            return containing[0]

        best_candidate = None
        best_distance = float("inf")
        for detection in self._latest_detections:
            cx, cy = detection.bbox.center
            distance = math.hypot(point[0] - cx, point[1] - cy)
            if distance <= self.click_config.search_radius and distance < best_distance:
                best_candidate = detection
                best_distance = distance
        return best_candidate

    def _find_candidate_by_engine_track_id(self) -> DetectedObject | None:
        if self._engine_track_id is None:
            return None
        for detection in self._latest_detections:
            if detection.track_id == self._engine_track_id:
                return detection
        return None

    def _find_reacquire_candidate(self) -> DetectedObject | None:
        if self._bbox is None:
            return None

        previous_bbox = self._bbox
        previous_cx, previous_cy = previous_bbox.center
        max_distance = (
            max(previous_bbox.width, previous_bbox.height) * self.neural_config.max_reacquire_distance_factor
            + self._lost_frames * self.config.lost_search_growth
        )
        best_candidate: DetectedObject | None = None
        best_score = float("-inf")

        for detection in self._latest_detections:
            if self.neural_config.prefer_same_class and self._target_class_id is not None:
                if detection.class_id is not None and detection.class_id != self._target_class_id:
                    continue

            candidate_cx, candidate_cy = detection.bbox.center
            distance = math.hypot(candidate_cx - previous_cx, candidate_cy - previous_cy)
            if distance > max_distance:
                continue

            iou = detection.bbox.intersection_over_union(previous_bbox)
            distance_score = max(0.0, 1.0 - distance / max(max_distance, 1.0))
            size_ratio = min(detection.bbox.area, previous_bbox.area) / max(detection.bbox.area, previous_bbox.area, 1)
            score = detection.confidence + 0.35 * iou + 0.25 * distance_score + 0.15 * size_ratio
            if score > best_score:
                best_score = score
                best_candidate = detection

        return best_candidate

    def _apply_candidate(self, candidate: DetectedObject, *, reacquired: bool) -> None:
        self._bbox = candidate.bbox
        self._predicted_bbox = candidate.bbox
        self._search_region = candidate.bbox
        self._engine_track_id = candidate.track_id
        self._target_class_id = candidate.class_id
        self._score = candidate.confidence
        self._lost_frames = 0
        self._state = TrackerState.TRACKING if not reacquired else TrackerState.TRACKING

    def _expand_search_region(self, frame_shape: tuple[int, int] | tuple[int, int, int]) -> BoundingBox | None:
        if self._bbox is None:
            return None
        padding = self.config.search_margin + self._lost_frames * self.config.lost_search_growth
        return self._bbox.pad(padding, padding).clamp(frame_shape)
