from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .....domain.models import BoundingBox, ProcessedFrame, TrackerState
from ....frame_stabilization import FrameStabilizerResult
from ....target_selection import TargetPolarity
from ....target_selection.manager import TargetSelectionManager
from ...result import TargetTrackingResult
from .search_result import SearchResult
from ...predictors import KalmanTargetPredictor
from ..base_target_tracker import BaseTargetTracker
from .bbox_stabilizer import BboxStabilizer
from .contrast_relocator import ContrastRelocator
from .edge_exit_guard import EdgeExitGuard
from .feature_point_tracker import FeaturePointTracker
from .frame_quality_monitor import FrameQualityMonitor
from .point_prediction import PointPrediction
from .template_matcher import TemplateMatcher
from .template_point_config import TemplatePointTargetTrackerConfig
from .template_storage import TemplateStorage


@dataclass(slots=True)
class TemplatePointTargetTracker(BaseTargetTracker):
    """Сопровождает одну цель по шаблонам, опорным точкам и прогнозу движения."""

    config: TemplatePointTargetTrackerConfig

    _selector: TargetSelectionManager = field(init=False, repr=False)
    _target_predictor: KalmanTargetPredictor = field(init=False, repr=False)
    _templates: TemplateStorage = field(init=False, repr=False)
    _feature_points: FeaturePointTracker = field(init=False, repr=False)
    _quality: FrameQualityMonitor = field(init=False, repr=False)
    _edge_guard: EdgeExitGuard = field(init=False, repr=False)
    _bbox_stabilizer: BboxStabilizer = field(init=False, repr=False)
    _matcher: TemplateMatcher = field(init=False, repr=False)
    _contrast_relocator: ContrastRelocator = field(init=False, repr=False)

    _next_track_id: int = field(default=0, init=False)
    _track_id: int | None = field(default=None, init=False)
    _state: TrackerState = field(default=TrackerState.IDLE, init=False)
    _bbox: BoundingBox | None = field(default=None, init=False)
    _predicted_bbox: BoundingBox | None = field(default=None, init=False)
    _search_region: BoundingBox | None = field(default=None, init=False)
    _lost_frames: int = field(default=0, init=False)
    _score: float = field(default=0.0, init=False)
    _message: str = field(default="Click target", init=False)

    _residual_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32),
        init=False,
        repr=False,
    )
    _camera_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32),
        init=False,
        repr=False,
    )
    _target_polarity: TargetPolarity = field(default=TargetPolarity.HOT, init=False)

    def __post_init__(self) -> None:
        """Подготовить внутренние компоненты template-point трекера."""
        self._selector = TargetSelectionManager((self.config.target_selector_config,))
        self._target_predictor = KalmanTargetPredictor()
        self._templates = TemplateStorage(config=self.config)
        self._feature_points = FeaturePointTracker(config=self.config)
        self._quality = FrameQualityMonitor(config=self.config)
        self._edge_guard = EdgeExitGuard(config=self.config)
        self._bbox_stabilizer = BboxStabilizer(config=self.config)
        self._matcher = TemplateMatcher(config=self.config, storage=self._templates)
        self._contrast_relocator = ContrastRelocator(
            config=self.config,
            selector=self._selector,
        )

    def reset(self) -> TargetTrackingResult:
        """Полностью сбросить состояние трекера."""
        self._track_id = None
        self._state = TrackerState.IDLE
        self._bbox = None
        self._predicted_bbox = None
        self._search_region = None
        self._lost_frames = 0
        self._score = 0.0
        self._message = "Tracker reset"
        self._residual_velocity[:] = 0.0
        self._camera_offset[:] = 0.0
        self._target_polarity = TargetPolarity.HOT

        self._target_predictor.reset()
        self._templates.reset()
        self._feature_points.reset()
        self._quality.reset()
        self._edge_guard.reset()

        return self.snapshot(FrameStabilizerResult())

    def start_tracking(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
    ) -> TargetTrackingResult:
        """Начать сопровождение новой цели по одному клику."""
        selection = self._selector.apply(frame=frame, point=point)

        if selection is None:
            return self.reset()

        bbox = selection.bbox.clamp(frame.bgr.shape)

        if not self._templates.initialize(frame=frame, bbox=bbox):
            return self.reset()

        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._bbox = bbox
        self._predicted_bbox = bbox
        self._search_region = bbox.pad(
            self.config.search_margin,
            self.config.search_margin,
        ).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._state = TrackerState.TRACKING
        self._score = 1.0
        self._message = f"Selected target #{self._track_id}"
        self._residual_velocity[:] = 0.0
        self._camera_offset[:] = 0.0
        self._target_polarity = ContrastRelocator.resolve_target_polarity(
            frame=frame,
            bbox=bbox,
            selection_polarity=selection.polarity,
        )

        self._target_predictor.initialize(self._to_predictor_bbox(bbox))
        self._quality.update_baseline(frame)
        self._feature_points.initialize(frame=frame, bbox=bbox, force=True)
        self._feature_points.set_previous_frame(frame)
        self._edge_guard.update_exit_edges(bbox=bbox, frame_shape=frame.bgr.shape)

        return self.snapshot(FrameStabilizerResult())

    def update(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
    ) -> TargetTrackingResult:
        """Обновить трекер на новом кадре."""
        if self._state == TrackerState.IDLE or self._bbox is None:
            self._message = "Click target"
            self._feature_points.set_previous_frame(frame)
            return self.snapshot(motion)

        frame_degraded = self._quality.update_state(frame)

        if not frame_degraded and not (
            self._state == TrackerState.SEARCHING and self._quality.blur_hold_active()
        ):
            self._update_camera_offset(motion)

        point_prediction = None if frame_degraded else self._feature_points.predict(
            frame=frame,
            bbox=self._bbox,
        )
        predicted_bbox = self._predict_bbox(
            frame_shape=frame.bgr.shape,
            motion=motion,
            point_prediction=point_prediction,
        )
        search_region = self._build_search_region(
            predicted_bbox=predicted_bbox,
            frame_shape=frame.bgr.shape,
            use_tight_margin=point_prediction is not None,
        )

        search = self._matcher.locate_target(
            frame=frame,
            predicted_bbox=predicted_bbox,
            search_region=search_region,
            point_prediction=point_prediction,
            target_polarity=self._target_polarity,
            state=self._state,
            lost_frames=self._lost_frames,
        )

        threshold = (
            self.config.track_threshold
            if self._state == TrackerState.TRACKING
            else self.config.reacquire_threshold
        )

        if search is None or search.score < threshold:
            contrast_search = self._contrast_relocator.locate_by_contrast(
                frame=frame,
                predicted_bbox=predicted_bbox,
                target_polarity=self._target_polarity,
                state=self._state,
                lost_frames=self._lost_frames,
                frame_degraded=frame_degraded,
            )

            if contrast_search is not None and (search is None or contrast_search.score > search.score):
                search = contrast_search

        if search is not None and search.score >= threshold:
            accepted_snapshot = self._try_accept_search_result(
                frame=frame,
                motion=motion,
                search=search,
                predicted_bbox=predicted_bbox,
                frame_degraded=frame_degraded,
            )

            if accepted_snapshot is not None:
                return accepted_snapshot

        return self._mark_lost(
            frame=frame,
            motion=motion,
            predicted_bbox=predicted_bbox,
            search=search,
            point_prediction=point_prediction,
            frame_degraded=frame_degraded,
        )

    def snapshot(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Вернуть текущее состояние для отрисовки и отладки."""
        visible_bbox = self._bbox if self._state == TrackerState.TRACKING else None

        return TargetTrackingResult(
            state=self._state,
            track_id=self._track_id,
            bbox=visible_bbox,
            predicted_bbox=self._predicted_bbox,
            search_region=self._search_region,
            score=self._score,
            lost_frames=self._lost_frames,
            global_motion=motion,
            message=self._message,
        )

    def resume_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        track_id: int,
    ) -> TargetTrackingResult:
        """Возобновить сопровождение по подтверждённому recovery bbox."""
        clamped = bbox.clamp(frame.bgr.shape)

        if not self._templates.initialize(frame=frame, bbox=clamped):
            return self.reset()

        self._track_id = track_id
        self._next_track_id = max(self._next_track_id, track_id + 1)
        self._bbox = clamped
        self._predicted_bbox = clamped
        self._search_region = clamped.pad(
            self.config.search_margin,
            self.config.search_margin,
        ).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._state = TrackerState.TRACKING
        self._score = 1.0
        self._message = f"Resumed target #{track_id}"
        self._residual_velocity[:] = 0.0
        self._camera_offset[:] = 0.0
        self._target_polarity = ContrastRelocator.resolve_target_polarity(
            frame=frame,
            bbox=clamped,
            selection_polarity=TargetPolarity.HOT,
        )

        self._target_predictor.initialize(self._to_predictor_bbox(clamped))
        self._quality.update_baseline(frame)
        self._feature_points.initialize(frame=frame, bbox=clamped, force=True)
        self._feature_points.set_previous_frame(frame)
        self._edge_guard.update_exit_edges(bbox=clamped, frame_shape=frame.bgr.shape)

        return self.snapshot(FrameStabilizerResult())

    def _try_accept_search_result(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
        search: SearchResult,
        predicted_bbox: BoundingBox,
        *,
        frame_degraded: bool,
    ) -> TargetTrackingResult | None:
        """Попробовать принять найденный bbox как новое положение цели."""
        refined = self._selector.refine(frame=frame, bbox=search.bbox)
        measured_bbox = refined.bbox if refined is not None else search.bbox
        measured_bbox = measured_bbox.clamp(frame.bgr.shape)

        measured_bbox = self._bbox_stabilizer.stabilize(
            measured_bbox=measured_bbox,
            previous_bbox=self._bbox,
            canonical_size=self._templates.canonical_size,
            lost_frames=self._lost_frames,
            frame_shape=frame.bgr.shape,
        )

        trusted_measurement = self._is_trusted_measurement(
            score=search.score,
            measured_bbox=measured_bbox,
            predicted_bbox=predicted_bbox,
            frame_degraded=frame_degraded,
        )

        if (
            self._is_invalid_motion_candidate(measured_bbox, predicted_bbox)
            or self._edge_guard.is_invalid_edge_candidate(measured_bbox, frame.bgr.shape)
            or (frame_degraded and not trusted_measurement)
            or (self._state == TrackerState.SEARCHING and not trusted_measurement)
        ):
            return None

        if trusted_measurement:
            if not frame_degraded:
                self._update_velocity(measured_bbox=measured_bbox, motion=motion)
                self._quality.update_baseline(frame)

            self._target_predictor.update(self._to_predictor_bbox(measured_bbox))

        self._bbox = measured_bbox
        self._predicted_bbox = predicted_bbox
        self._search_region = search.search_region
        self._lost_frames = 0
        self._score = search.score
        self._state = TrackerState.TRACKING
        self._message = (
            f"Holding target #{self._track_id} through degraded frame"
            if frame_degraded
            else f"Tracking target #{self._track_id}"
        )
        self._edge_guard.update_exit_edges(bbox=measured_bbox, frame_shape=frame.bgr.shape)

        if not frame_degraded and self._templates.can_update(measured_bbox, search.score):
            self._templates.update(frame=frame, bbox=measured_bbox)

        if frame_degraded:
            self._feature_points.tracked_points = None
        else:
            self._feature_points.refresh(frame=frame, bbox=measured_bbox)

        self._feature_points.set_previous_frame(frame)

        return self.snapshot(motion)

    def _mark_lost(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
        predicted_bbox: BoundingBox,
        search: SearchResult | None,
        point_prediction: PointPrediction | None,
        *,
        frame_degraded: bool,
    ) -> TargetTrackingResult:
        """Обработать кадр, где цель не была надёжно найдена."""
        self._lost_frames += 1
        self._predicted_bbox = predicted_bbox
        self._search_region = self._build_search_region(
            predicted_bbox=predicted_bbox,
            frame_shape=frame.bgr.shape,
            use_tight_margin=point_prediction is not None,
        )
        self._score = search.score if search is not None else 0.0

        if self._edge_guard.has_exit_edges and self._lost_frames > self.config.edge_exit_max_lost_frames:
            return self._go_idle(
                frame=frame,
                motion=motion,
                message="Target left frame, select target again",
            )

        if self._lost_frames > self._quality.current_max_lost_frames():
            return self._go_idle(
                frame=frame,
                motion=motion,
                message="Target lost, select target again",
            )

        self._state = TrackerState.SEARCHING
        self._message = (
            f"Frame degraded, holding prediction #{self._track_id}"
            if frame_degraded
            else f"Searching for target #{self._track_id}"
        )
        self._feature_points.set_previous_frame(frame)

        return self.snapshot(motion)

    def _go_idle(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
        message: str,
    ) -> TargetTrackingResult:
        """Перевести трекер в IDLE после потери цели."""
        self._message = message
        self._state = TrackerState.IDLE
        self._bbox = None
        self._feature_points.tracked_points = None
        self._edge_guard.reset()
        self._feature_points.set_previous_frame(frame)

        return self.snapshot(motion)

    def _predict_bbox(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        motion: FrameStabilizerResult,
        point_prediction: PointPrediction | None = None,
    ) -> BoundingBox:
        """Построить прогноз bbox по predictor, точкам и сдвигу камеры."""
        assert self._bbox is not None

        predictor_prediction = self._target_predictor.predict()

        if predictor_prediction is not None:
            predicted = self._from_predictor_bbox(
                bbox=predictor_prediction,
                frame_shape=frame_shape,
            )

            if point_prediction is not None:
                predicted = self._blend_with_point_prediction(
                    predicted=predicted,
                    point_prediction=point_prediction,
                    frame_shape=frame_shape,
                )

            return predicted

        previous_center = np.array(self._bbox.center, dtype=np.float32)

        if motion.valid:
            motion_shift = np.array([motion.dx, motion.dy], dtype=np.float32)
        else:
            motion_shift = np.zeros(2, dtype=np.float32)

        predicted_center = previous_center + motion_shift + self._residual_velocity

        return BoundingBox.from_center(
            predicted_center[0],
            predicted_center[1],
            self._bbox.width,
            self._bbox.height,
        ).clamp(frame_shape)

    def _blend_with_point_prediction(
        self,
        predicted: BoundingBox,
        point_prediction: PointPrediction,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Смешать прогноз predictor с прогнозом по опорным точкам."""
        predicted_center = np.array(predicted.center, dtype=np.float32)
        point_center = np.array(point_prediction.bbox.center, dtype=np.float32)
        max_dimension = max(predicted.width, predicted.height, self.config.min_box_size)

        if float(np.linalg.norm(point_center - predicted_center)) > max_dimension * 1.8:
            return predicted

        blended_center = predicted_center * 0.72 + point_center * 0.28

        return BoundingBox.from_center(
            blended_center[0],
            blended_center[1],
            predicted.width,
            predicted.height,
        ).clamp(frame_shape)

    def _build_search_region(
        self,
        predicted_bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        use_tight_margin: bool,
    ) -> BoundingBox:
        """Построить область поиска вокруг прогнозного bbox."""
        if self._lost_frames >= self.config.full_frame_after:
            frame_height, frame_width = frame_shape[:2]
            return BoundingBox(0, 0, frame_width, frame_height)

        if use_tight_margin and self._lost_frames == 0:
            margin = self.config.point_search_margin
        else:
            margin = self.config.search_margin + self._lost_frames * self.config.lost_search_growth

        return predicted_bbox.pad(margin, margin).clamp(frame_shape)

    def _is_invalid_motion_candidate(
        self,
        bbox: BoundingBox,
        predicted_bbox: BoundingBox,
    ) -> bool:
        """Отсечь кандидата, который ломает набранную траекторию."""
        if self._bbox is None:
            return False

        predicted_center = np.array(predicted_bbox.center, dtype=np.float32)
        measured_center = np.array(bbox.center, dtype=np.float32)
        center_error = float(np.linalg.norm(measured_center - predicted_center))
        max_dimension = max(predicted_bbox.width, predicted_bbox.height, self.config.min_box_size)

        if self._lost_frames == 0:
            allowed_error = max(8.0, max_dimension * 1.25)
        else:
            growth = 0.04 * min(self._lost_frames, 20)

            if self._quality.blur_hold_active():
                growth += self.config.blur_hold_center_growth * min(self._lost_frames, 30)

            allowed_error = max(12.0, max_dimension * (1.35 + growth))

        return center_error > allowed_error

    def _is_trusted_measurement(
        self,
        score: float,
        measured_bbox: BoundingBox,
        predicted_bbox: BoundingBox,
        *,
        frame_degraded: bool,
    ) -> bool:
        """Разрешить обновление траектории только по надёжному совпадению."""
        predicted_center = np.array(predicted_bbox.center, dtype=np.float32)
        measured_center = np.array(measured_bbox.center, dtype=np.float32)
        center_error = float(np.linalg.norm(measured_center - predicted_center))
        max_dimension = max(predicted_bbox.width, predicted_bbox.height, self.config.min_box_size)

        if frame_degraded:
            strict_score = max(
                self.config.template_update_threshold + 0.12,
                self.config.reacquire_threshold + 0.20,
            )

            if self._state == TrackerState.TRACKING:
                allowed_error = max(5.0, max_dimension * 0.8)
            else:
                allowed_error = max(7.0, max_dimension * 1.1)

            return score >= strict_score and center_error <= allowed_error

        if score >= self.config.template_update_threshold:
            if self._state == TrackerState.SEARCHING:
                if self._quality.blur_hold_active():
                    allowed_error = max(
                        12.0,
                        max_dimension * (1.35 + 0.03 * min(self._lost_frames, 10)),
                    )

                    if center_error > allowed_error:
                        strong_score = self.config.template_update_threshold + 0.18
                        strong_error = max(18.0, max_dimension * 1.9)
                        return score >= strong_score and center_error <= strong_error

                    return True

                growth = 0.03 * min(self._lost_frames, 20)
                allowed_error = max(10.0, max_dimension * (1.45 + growth))
                return center_error <= allowed_error

            return True

        close_to_prediction = center_error <= max(6.0, max_dimension * 0.75)
        almost_confident = score >= self.config.track_threshold + 0.04

        return self._state == TrackerState.TRACKING and close_to_prediction and almost_confident

    def _update_camera_offset(self, motion: FrameStabilizerResult) -> None:
        """Накопить сдвиг камеры с момента выбора текущей цели."""
        if not motion.valid:
            return

        self._camera_offset += np.array([motion.dx, motion.dy], dtype=np.float32)

    def _to_predictor_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """Перевести bbox в координаты predictor с компенсацией камеры."""
        center = np.array(bbox.center, dtype=np.float32) - self._camera_offset

        return BoundingBox.from_center(
            center[0],
            center[1],
            bbox.width,
            bbox.height,
        )

    def _from_predictor_bbox(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Вернуть прогноз predictor обратно в координаты текущего кадра."""
        assert self._bbox is not None

        center = np.array(bbox.center, dtype=np.float32) + self._camera_offset

        return BoundingBox.from_center(
            center[0],
            center[1],
            self._bbox.width,
            self._bbox.height,
        ).clamp(frame_shape)

    def _update_velocity(
        self,
        measured_bbox: BoundingBox,
        motion: FrameStabilizerResult,
    ) -> None:
        """Обновить остаточную скорость цели относительно движения камеры."""
        assert self._bbox is not None

        previous_center = np.array(self._bbox.center, dtype=np.float32)

        if motion.valid:
            motion_shift = np.array([motion.dx, motion.dy], dtype=np.float32)
        else:
            motion_shift = np.zeros(2, dtype=np.float32)

        baseline_center = previous_center + motion_shift
        measured_center = np.array(measured_bbox.center, dtype=np.float32)
        new_velocity = measured_center - baseline_center
        alpha = self.config.velocity_alpha
        self._residual_velocity = self._residual_velocity * (1.0 - alpha) + new_velocity * alpha
