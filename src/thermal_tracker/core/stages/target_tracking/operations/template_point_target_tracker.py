"""Трекер одной выбранной цели.

Здесь живёт основной рабочий движок:
- оператор кликает по цели;
- цель автоматически выделяется вокруг точки;
- дальше трекер ведёт именно её;
- если цель пропала, трекер пытается найти её снова с тем же ID.

Подход гибридный:
- шаблоны отвечают за внешний вид цели;
- опорные точки отвечают за локальное движение;
- глобальное движение камеры учитывается отдельно.
"""

from __future__ import annotations

import cv2
import numpy as np

from thermal_tracker.core.config import ClickSelectionConfig, OpenCVTrackerConfig
from thermal_tracker.core.domain.models import BoundingBox, ProcessedFrame, TrackSnapshot, TrackerState
from thermal_tracker.core.stages.frame_stabilization.result import FrameStabilizerResult
from thermal_tracker.core.stages.target_selection import TargetSelectorManager
from .base_target_tracker import BaseSingleTargetTracker
from thermal_tracker.core.stages.target_tracking.motion_models import KalmanMotionModel
from thermal_tracker.core.stages.target_tracking.point_prediction import PointPrediction
from thermal_tracker.core.stages.target_tracking.result import SearchResult


def _safe_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Меняет размер изображения и не даёт получить нулевую ширину или высоту."""
    width, height = size
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    """Вырезает прямоугольник из изображения и возвращает `None`, если он выродился."""
    clamped = bbox.clamp(image.shape)
    if clamped.width <= 1 or clamped.height <= 1:
        return None
    return image[clamped.y:clamped.y2, clamped.x:clamped.x2]


def _correlation(image: np.ndarray, template: np.ndarray) -> float:
    """Считает нормализованную корреляцию двух одинаковых по размеру патчей."""
    if image.shape != template.shape:
        return 0.0
    if float(np.std(image)) < 1e-6 or float(np.std(template)) < 1e-6:
        return 0.0
    score_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return float(score_map[0, 0])


class ClickToTrackSingleTargetTracker(BaseSingleTargetTracker):
    """Трекер одной цели по клику.
    Логика простая:
    - после клика выбираем объект вокруг точки;
    - сохраняем два шаблона: долгий и адаптивный;
    - в каждом кадре сначала пробуем понять движение по точкам;
    - затем ищем лучший кандидат по внешнему виду;
    - если цель пропала, расширяем область поиска и пытаемся вернуть тот же ID.
    """

    def __init__(self, tracker_config: OpenCVTrackerConfig, click_config: ClickSelectionConfig) -> None:
        self.config = tracker_config
        self.selector = TargetSelectorManager(click_config.method, click_config)
        self._next_track_id = 0
        self._track_id: int | None = None
        self._state: TrackerState = TrackerState.IDLE
        self._bbox: BoundingBox | None = None
        self._predicted_bbox: BoundingBox | None = None
        self._search_region: BoundingBox | None = None
        self._lost_frames = 0
        self._score = 0.0
        self._message = "Click target"
        self._residual_velocity = np.zeros(2, dtype=np.float32)
        self._canonical_size: tuple[int, int] | None = None
        self._long_term_gray: np.ndarray | None = None
        self._long_term_grad: np.ndarray | None = None
        self._adaptive_gray: np.ndarray | None = None
        self._adaptive_grad: np.ndarray | None = None
        self._previous_normalized: np.ndarray | None = None
        self._tracked_points: np.ndarray | None = None
        self._target_polarity = "unknown"
        self._point_center_offset = np.zeros(2, dtype=np.float32)
        self._frames_since_feature_refresh = 0
        self._exit_edges: set[str] = set()
        self._motion_model = KalmanMotionModel()
        self._camera_offset = np.zeros(2, dtype=np.float32)
        self._sharpness_baseline: float | None = None
        self._degraded_frames = 0
        self._blur_hold_frames = 0

    def reset(self) -> TrackSnapshot:
        """Полностью сбрасывает состояние трекера."""
        self._track_id = None
        self._state = TrackerState.IDLE
        self._bbox = None
        self._predicted_bbox = None
        self._search_region = None
        self._lost_frames = 0
        self._score = 0.0
        self._message = "Tracker reset"
        self._residual_velocity[:] = 0.0
        self._canonical_size = None
        self._long_term_gray = None
        self._long_term_grad = None
        self._adaptive_gray = None
        self._adaptive_grad = None
        self._previous_normalized = None
        self._tracked_points = None
        self._target_polarity = "unknown"
        self._point_center_offset[:] = 0.0
        self._frames_since_feature_refresh = 0
        self._exit_edges.clear()
        self._motion_model.reset()
        self._camera_offset[:] = 0.0
        self._sharpness_baseline = None
        self._degraded_frames = 0
        self._blur_hold_frames = 0
        return self.snapshot(FrameStabilizerResult())

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает сопровождение новой цели по одному клику."""
        selection = self.selector.select(frame, point)
        bbox = selection.bbox.clamp(frame.bgr.shape)
        canonical_w = max(self.config.min_box_size, bbox.width)
        canonical_h = max(self.config.min_box_size, bbox.height)
        self._canonical_size = (canonical_w, canonical_h)

        gray_patch = _crop(frame.normalized, bbox)
        grad_patch = _crop(frame.gradient, bbox)
        if gray_patch is None or grad_patch is None:
            return self.reset()

        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._bbox = bbox
        self._predicted_bbox = bbox
        self._search_region = bbox.pad(self.config.search_margin, self.config.search_margin).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._state = TrackerState.TRACKING
        self._score = 1.0
        self._message = f"Selected target #{self._track_id}"
        self._residual_velocity[:] = 0.0
        self._camera_offset[:] = 0.0
        self._target_polarity = self._resolve_target_polarity(frame, bbox, selection.polarity)
        self._motion_model.initialize(self._to_motion_model_bbox(bbox))
        self._sharpness_baseline = self._measure_frame_sharpness(frame)
        self._degraded_frames = 0
        self._blur_hold_frames = 0

        self._long_term_gray = _safe_resize(gray_patch, self._canonical_size)
        self._long_term_grad = _safe_resize(grad_patch, self._canonical_size)
        self._adaptive_gray = self._long_term_gray.copy()
        self._adaptive_grad = self._long_term_grad.copy()

        # Сразу набираем точки на объекте, чтобы не ехать только на шаблоне.
        self._initialize_feature_points(frame, bbox, force=True)
        self._previous_normalized = frame.normalized.copy()
        self._update_exit_edges(bbox, frame.bgr.shape)
        return self.snapshot(FrameStabilizerResult())

    def update(self, frame: ProcessedFrame, global_motion: FrameStabilizerResult) -> TrackSnapshot:
        """Обновляет трекер на новом кадре."""
        if self._state == TrackerState.IDLE or self._bbox is None:
            self._message = "Click target"
            self._previous_normalized = frame.normalized.copy()
            return self.snapshot(global_motion)

        frame_degraded = self._update_frame_quality_state(frame)
        if not frame_degraded and not (self._state == TrackerState.SEARCHING and self._blur_hold_active()):
            self._update_camera_offset(global_motion)
        point_prediction = None if frame_degraded else self._predict_from_points(frame)
        predicted_bbox = self._predict_bbox(frame.bgr.shape, global_motion, point_prediction)
        search = self._locate_target(frame, predicted_bbox, point_prediction)
        threshold = self.config.track_threshold if self._state == TrackerState.TRACKING else self.config.reacquire_threshold
        if search is None or search.score < threshold:
            contrast_search = self._locate_by_contrast(frame, predicted_bbox, frame_degraded=frame_degraded)
            if contrast_search is not None and (search is None or contrast_search.score > search.score):
                search = contrast_search

        if search is not None and search.score >= threshold:
            refined = self.selector.refine(frame, search.bbox)
            measured_bbox = refined.bbox if refined is not None else search.bbox
            measured_bbox = measured_bbox.clamp(frame.bgr.shape)
            measured_bbox = self._stabilize_bbox_size(measured_bbox, frame.bgr.shape)
            trusted_measurement = self._is_trusted_measurement(
                search.score,
                measured_bbox,
                predicted_bbox,
                frame_degraded=frame_degraded,
            )
            if (
                self._is_invalid_motion_candidate(measured_bbox, predicted_bbox)
                or self._is_invalid_edge_candidate(
                    measured_bbox,
                    frame.bgr.shape,
                )
                or (frame_degraded and not trusted_measurement)
                or (self._state == TrackerState.SEARCHING and not trusted_measurement)
            ):
                search = SearchResult(bbox=measured_bbox, score=-1.0, search_region=search.search_region)
            else:
                if trusted_measurement:
                    if not frame_degraded:
                        self._update_velocity(measured_bbox, global_motion)
                        self._update_sharpness_baseline(frame)
                    self._motion_model.update(self._to_motion_model_bbox(measured_bbox))
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
                self._update_exit_edges(measured_bbox, frame.bgr.shape)

                if not frame_degraded and self._can_update_templates(measured_bbox, search.score):
                    self._update_templates(frame, measured_bbox)

                if frame_degraded:
                    self._tracked_points = None
                else:
                    self._refresh_feature_points(frame, measured_bbox)
                self._previous_normalized = frame.normalized.copy()
                return self.snapshot(global_motion)

        self._lost_frames += 1
        self._predicted_bbox = predicted_bbox
        self._search_region = self._build_search_region(
            predicted_bbox,
            frame.bgr.shape,
            point_prediction is not None,
        )
        self._score = search.score if search is not None else 0.0

        if self._exit_edges and self._lost_frames > self.config.edge_exit_max_lost_frames:
            self._message = "Target left frame, contrast_component again"
            self._state = TrackerState.IDLE
            self._bbox = None
            self._tracked_points = None
            self._exit_edges.clear()
            self._previous_normalized = frame.normalized.copy()
            return self.snapshot(global_motion)

        if self._lost_frames > self._current_max_lost_frames():
            self._message = "Target lost, contrast_component again"
            self._state = TrackerState.IDLE
            self._bbox = None
            self._tracked_points = None
            self._exit_edges.clear()
            self._previous_normalized = frame.normalized.copy()
            return self.snapshot(global_motion)

        self._state = TrackerState.SEARCHING
        self._message = (
            f"Frame degraded, holding prediction #{self._track_id}"
            if frame_degraded
            else f"Searching for target #{self._track_id}"
        )
        self._previous_normalized = frame.normalized.copy()
        return self.snapshot(global_motion)

    def snapshot(self, global_motion: FrameStabilizerResult) -> TrackSnapshot:
        """Возвращает текущее состояние для отрисовки и отладки."""
        visible_bbox = self._bbox if self._state == TrackerState.TRACKING else None
        return TrackSnapshot(
            state=self._state,
            track_id=self._track_id,
            bbox=visible_bbox,
            predicted_bbox=self._predicted_bbox,
            search_region=self._search_region,
            score=self._score,
            lost_frames=self._lost_frames,
            global_motion=global_motion,
            message=self._message,
        )

    def _predict_bbox(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        global_motion: FrameStabilizerResult,
        point_prediction: PointPrediction | None = None,
    ) -> BoundingBox:
        """Строит прогноз по модели движения, точкам и сдвигу камеры."""
        assert self._bbox is not None

        motion_prediction = self._motion_model.predict()
        if motion_prediction is not None:
            predicted = self._from_motion_model_bbox(motion_prediction, frame_shape)
            if point_prediction is not None:
                predicted_center = np.array(predicted.center, dtype=np.float32)
                point_center = np.array(point_prediction.bbox.center, dtype=np.float32)
                max_dimension = max(predicted.width, predicted.height, self.config.min_box_size)
                if float(np.linalg.norm(point_center - predicted_center)) <= max_dimension * 1.8:
                    blended_center = predicted_center * 0.72 + point_center * 0.28
                    predicted = BoundingBox.from_center(
                        blended_center[0],
                        blended_center[1],
                        predicted.width,
                        predicted.height,
                    ).clamp(frame_shape)
            return predicted

        previous_center = np.array(self._bbox.center, dtype=np.float32)
        motion_shift = np.array([global_motion.dx, global_motion.dy], dtype=np.float32) if global_motion.valid else 0.0
        predicted_center = previous_center + motion_shift + self._residual_velocity
        predicted = BoundingBox.from_center(
            predicted_center[0],
            predicted_center[1],
            self._bbox.width,
            self._bbox.height,
        )
        return predicted.clamp(frame_shape)

    def _predict_from_points(self, frame: ProcessedFrame) -> PointPrediction | None:
        """Пробует спрогнозировать движение цели по опорным точкам внутри объекта."""
        if self._previous_normalized is None or self._tracked_points is None or self._bbox is None:
            return None

        if len(self._tracked_points) < self.config.min_feature_points:
            self._tracked_points = None
            return None

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._previous_normalized,
            frame.normalized,
            self._tracked_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        if next_points is None or status is None:
            self._tracked_points = None
            return None

        valid_mask = status.flatten() == 1
        if errors is not None:
            valid_mask &= errors.flatten() < 25.0

        previous_points = self._tracked_points.reshape(-1, 2)[valid_mask]
        current_points = next_points.reshape(-1, 2)[valid_mask]
        if len(current_points) < self.config.min_feature_points:
            self._tracked_points = None
            return None

        displacements = current_points - previous_points
        median_displacement = np.median(displacements, axis=0)
        residuals = np.linalg.norm(displacements - median_displacement, axis=1)
        allowed_residual = max(3.0, float(np.median(residuals) * 2.5 + 1.0))
        consistent_mask = residuals <= allowed_residual
        current_points = current_points[consistent_mask]
        if len(current_points) < self.config.min_feature_points:
            self._tracked_points = None
            return None

        self._tracked_points = current_points.reshape(-1, 1, 2).astype(np.float32)
        point_center = np.median(current_points, axis=0) + self._point_center_offset
        predicted_bbox = BoundingBox.from_center(
            point_center[0],
            point_center[1],
            self._bbox.width,
            self._bbox.height,
        ).clamp(frame.bgr.shape)

        confidence = min(1.0, len(current_points) / max(float(self.config.max_feature_points), 1.0))
        return PointPrediction(bbox=predicted_bbox, confidence=confidence)

    def _build_search_region(
        self,
        predicted_bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        use_tight_margin: bool,
    ) -> BoundingBox:
        """Строит область поиска.

        Если точки хорошо живут, ищем ближе к прогнозу.
        Если цель теряется, постепенно расширяемся.
        """
        if self._lost_frames >= self.config.full_frame_after:
            frame_h, frame_w = frame_shape[:2]
            return BoundingBox(0, 0, frame_w, frame_h)

        if use_tight_margin and self._lost_frames == 0:
            margin = self.config.point_search_margin
        else:
            margin = self.config.search_margin + self._lost_frames * self.config.lost_search_growth
        return predicted_bbox.pad(margin, margin).clamp(frame_shape)

    def _locate_target(
        self,
        frame: ProcessedFrame,
        predicted_bbox: BoundingBox,
        point_prediction: PointPrediction | None,
    ) -> SearchResult | None:
        """Ищет лучший кандидат цели на кадре."""
        if self._canonical_size is None or self._adaptive_gray is None or self._long_term_gray is None:
            return None

        search_region = self._build_search_region(
            predicted_bbox,
            frame.bgr.shape,
            point_prediction is not None,
        )
        search_gray = _crop(frame.normalized, search_region)
        if search_gray is None:
            return None

        best_result: SearchResult | None = None
        for scale in self.config.scales:
            candidate_width = max(self.config.min_box_size, int(round(predicted_bbox.width * scale)))
            candidate_height = max(self.config.min_box_size, int(round(predicted_bbox.height * scale)))
            if candidate_width >= search_region.width or candidate_height >= search_region.height:
                continue

            adaptive_template = _safe_resize(self._adaptive_gray, (candidate_width, candidate_height))
            long_term_template = _safe_resize(self._long_term_gray, (candidate_width, candidate_height))
            candidate = self._evaluate_template_pair(
                frame=frame,
                search_region=search_region,
                search_gray=search_gray,
                template_gray=adaptive_template,
                fallback_template=long_term_template,
                candidate_width=candidate_width,
                candidate_height=candidate_height,
                predicted_bbox=predicted_bbox,
                point_prediction=point_prediction,
            )
            if candidate is None:
                continue

            if best_result is None or candidate.score > best_result.score:
                best_result = candidate

        return best_result

    def _locate_by_contrast(
        self,
        frame: ProcessedFrame,
        predicted_bbox: BoundingBox,
        *,
        frame_degraded: bool,
    ) -> SearchResult | None:
        """Ищет яркую/холодную компоненту рядом с прогнозом, когда шаблон не сработал."""

        point = (int(predicted_bbox.center[0]), int(predicted_bbox.center[1]))
        selection = self.selector.select(frame, point, expected_bbox=predicted_bbox)
        if selection.confidence < 0.7:
            return None
        if self._target_polarity in {"hot", "cold"} and selection.polarity != self._target_polarity:
            return None

        candidate_bbox = selection.bbox.clamp(frame.bgr.shape)
        candidate_center = np.array(candidate_bbox.center, dtype=np.float32)
        predicted_center = np.array(predicted_bbox.center, dtype=np.float32)
        center_error = float(np.linalg.norm(candidate_center - predicted_center))
        max_dimension = max(predicted_bbox.width, predicted_bbox.height, self.config.min_box_size)
        if self._state == TrackerState.TRACKING:
            allowed_error = max(8.0, max_dimension * 0.95)
        else:
            allowed_error = max(12.0, max_dimension * (1.15 + 0.05 * min(self._lost_frames, 12)))
        if center_error > allowed_error:
            return None

        width_ratio = candidate_bbox.width / max(float(predicted_bbox.width), 1.0)
        height_ratio = candidate_bbox.height / max(float(predicted_bbox.height), 1.0)
        if max(width_ratio, height_ratio) > 1.8 or min(width_ratio, height_ratio) < 0.45:
            return None

        search_region = self._build_search_region(predicted_bbox, frame.bgr.shape, use_tight_margin=False)
        score = (
            max(self.config.template_update_threshold + 0.14, self.config.reacquire_threshold + 0.22)
            if frame_degraded
            else min(self.config.template_update_threshold - 0.02, self.config.track_threshold + 0.18)
        )
        return SearchResult(bbox=candidate_bbox, score=score, search_region=search_region)

    def _evaluate_template_pair(
        self,
        frame: ProcessedFrame,
        search_region: BoundingBox,
        search_gray: np.ndarray,
        template_gray: np.ndarray,
        fallback_template: np.ndarray,
        candidate_width: int,
        candidate_height: int,
        predicted_bbox: BoundingBox,
        point_prediction: PointPrediction | None,
    ) -> SearchResult | None:
        """Проверяет кандидатов по двум шаблонам: адаптивному и долгому."""
        response_primary = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        response_fallback = cv2.matchTemplate(search_gray, fallback_template, cv2.TM_CCOEFF_NORMED)

        candidates = []
        for response in (response_primary, response_fallback):
            candidates.extend(self._collect_template_candidates(response, candidate_width, candidate_height))

        best_score = -1.0
        best_bbox: BoundingBox | None = None
        for template_score, max_location in candidates:
            candidate_bbox = BoundingBox(
                x=search_region.x + int(max_location[0]),
                y=search_region.y + int(max_location[1]),
                width=candidate_width,
                height=candidate_height,
            ).clamp(frame.bgr.shape)
            score = self._score_candidate(
                frame=frame,
                candidate_bbox=candidate_bbox,
                predicted_bbox=predicted_bbox,
                template_score=template_score,
                search_region=search_region,
                point_prediction=point_prediction,
            )
            if score > best_score:
                best_score = score
                best_bbox = candidate_bbox

        if best_bbox is None:
            return None

        return SearchResult(bbox=best_bbox, score=best_score, search_region=search_region)

    def _collect_template_candidates(
        self,
        response: np.ndarray,
        candidate_width: int,
        candidate_height: int,
        limit: int = 8,
    ) -> list[tuple[float, tuple[int, int]]]:
        """Берёт несколько локальных максимумов, а не только самый яркий ответ шаблона."""

        work = response.copy()
        candidates: list[tuple[float, tuple[int, int]]] = []
        suppression_radius = max(3, min(candidate_width, candidate_height) // 2)
        for _ in range(limit):
            _, max_value, _, max_location = cv2.minMaxLoc(work)
            if float(max_value) <= 0.05:
                break
            candidates.append((float(max_value), max_location))
            x, y = max_location
            left = max(0, x - suppression_radius)
            right = min(work.shape[1], x + suppression_radius + 1)
            top = max(0, y - suppression_radius)
            bottom = min(work.shape[0], y + suppression_radius + 1)
            work[top:bottom, left:right] = -1.0
        return candidates

    def _score_candidate(
        self,
        frame: ProcessedFrame,
        candidate_bbox: BoundingBox,
        predicted_bbox: BoundingBox,
        template_score: float,
        search_region: BoundingBox,
        point_prediction: PointPrediction | None,
    ) -> float:
        """Считает итоговый балл кандидата.

        Чем выше балл, тем больше похоже, что это именно наша цель.
        """
        assert self._canonical_size is not None
        assert self._adaptive_gray is not None and self._adaptive_grad is not None
        assert self._long_term_gray is not None and self._long_term_grad is not None

        candidate_gray = _crop(frame.normalized, candidate_bbox)
        candidate_grad = _crop(frame.gradient, candidate_bbox)
        if candidate_gray is None or candidate_grad is None:
            return -1.0

        normalized_gray = _safe_resize(candidate_gray, self._canonical_size)
        normalized_grad = _safe_resize(candidate_grad, self._canonical_size)
        gray_adaptive = _correlation(normalized_gray, self._adaptive_gray)
        gray_long = _correlation(normalized_gray, self._long_term_gray)
        grad_adaptive = _correlation(normalized_grad, self._adaptive_grad)
        grad_long = _correlation(normalized_grad, self._long_term_grad)
        contrast_score = self._local_contrast_score(frame, candidate_bbox)

        predicted_center = np.array(predicted_bbox.center, dtype=np.float32)
        candidate_center = np.array(candidate_bbox.center, dtype=np.float32)
        center_distance = float(np.linalg.norm(candidate_center - predicted_center))
        search_scale = max(search_region.width, search_region.height, 1)
        distance_penalty = (center_distance / search_scale) * self.config.distance_penalty
        max_dimension = max(predicted_bbox.width, predicted_bbox.height, self.config.min_box_size)

        if point_prediction is not None:
            point_center = np.array(point_prediction.bbox.center, dtype=np.float32)
            point_distance = float(np.linalg.norm(candidate_center - point_center))
            point_penalty = (point_distance / search_scale) * self.config.distance_penalty * (0.8 + point_prediction.confidence)
            if self._lost_frames == 0 and point_distance > max(predicted_bbox.width, predicted_bbox.height) * 1.4:
                return -1.0
        else:
            point_penalty = 0.0
            if self._lost_frames == 0:
                max_center_distance = max_dimension * self.config.max_tracking_center_shift
            else:
                max_center_distance = max_dimension * (
                    self.config.max_reacquire_center_shift
                    + self.config.reacquire_center_growth * min(self._lost_frames, 8)
                )
            if center_distance > max_center_distance and self._lost_frames < self.config.full_frame_after:
                return -1.0

        combined = (
            template_score * 0.24
            + gray_adaptive * 0.27
            + gray_long * 0.16
            + grad_adaptive * 0.14
            + grad_long * 0.07
            + contrast_score * 0.12
        )
        return combined - distance_penalty - point_penalty

    def _resolve_target_polarity(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        selection_polarity: str,
    ) -> str:
        """Определяет, цель горячее или холоднее локального фона."""

        if selection_polarity in {"hot", "cold"}:
            return selection_polarity

        levels = self._measure_local_contrast_levels(frame, bbox)
        if levels is None:
            return "hot"

        object_hot_level, object_cold_level, background_level = levels
        hot_contrast = object_hot_level - background_level
        cold_contrast = background_level - object_cold_level
        return "hot" if hot_contrast >= cold_contrast else "cold"

    def _local_contrast_score(self, frame: ProcessedFrame, bbox: BoundingBox) -> float:
        """Оценивает, насколько кандидат отделяется от локального фона."""

        levels = self._measure_local_contrast_levels(frame, bbox)
        if levels is None:
            return -0.2

        object_hot_level, object_cold_level, background_level = levels
        if self._target_polarity == "cold":
            contrast = background_level - object_cold_level
        else:
            contrast = object_hot_level - background_level

        min_contrast = max(float(self.selector.config.min_object_contrast), 1.0)
        normalized = (contrast - min_contrast * 0.5) / (min_contrast * 4.0)
        return float(np.clip(normalized, -1.0, 1.0))

    def _measure_local_contrast_levels(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
    ) -> tuple[float, float, float] | None:
        """Измеряет яркость кандидата и фона в кольце вокруг него."""

        image = frame.normalized
        clamped = bbox.clamp(image.shape)
        object_patch = image[clamped.y:clamped.y2, clamped.x:clamped.x2]
        if object_patch.size == 0:
            return None

        margin = max(4, min(18, int(round(max(clamped.width, clamped.height) * 1.25))))
        outer = clamped.pad(margin, margin).clamp(image.shape)
        ring_patch = image[outer.y:outer.y2, outer.x:outer.x2]
        if ring_patch.size == 0:
            return None

        ring_mask = np.ones(ring_patch.shape, dtype=bool)
        inner_x1 = clamped.x - outer.x
        inner_y1 = clamped.y - outer.y
        inner_x2 = inner_x1 + clamped.width
        inner_y2 = inner_y1 + clamped.height
        ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
        background_values = ring_patch[ring_mask]
        if background_values.size < 8:
            return None

        object_values = object_patch.reshape(-1)
        object_hot_level = float(np.percentile(object_values, 75))
        object_cold_level = float(np.percentile(object_values, 25))
        background_level = float(np.median(background_values))
        return object_hot_level, object_cold_level, background_level

    def _measure_frame_sharpness(self, frame: ProcessedFrame) -> float:
        """Оценивает резкость центральной части кадра без служебных надписей по краям."""

        gray = frame.gray
        frame_h, frame_w = gray.shape[:2]
        x1 = int(round(frame_w * 0.08))
        x2 = int(round(frame_w * 0.92))
        y1 = int(round(frame_h * 0.18))
        y2 = int(round(frame_h * 0.82))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            roi = gray

        laplacian = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
        return float(np.percentile(np.abs(laplacian), 90))

    def _update_frame_quality_state(self, frame: ProcessedFrame) -> bool:
        """Определяет, похож ли текущий кадр на замутнённый, и включает удержание прогноза."""

        if not self.config.blur_hold_enabled:
            self._degraded_frames = 0
            self._blur_hold_frames = 0
            return False

        sharpness = self._measure_frame_sharpness(frame)
        if self._sharpness_baseline is None:
            self._sharpness_baseline = max(sharpness, 1e-6)
            return False

        baseline = max(self._sharpness_baseline, 1e-6)
        degraded = sharpness <= baseline * self.config.blur_sharpness_drop_ratio
        if degraded:
            self._degraded_frames += 1
            self._blur_hold_frames = max(self._blur_hold_frames, self.config.blur_hold_max_frames)
        else:
            self._degraded_frames = 0
            if self._blur_hold_frames > 0:
                self._blur_hold_frames -= 1
        return degraded

    def _update_sharpness_baseline(self, frame: ProcessedFrame) -> None:
        """Обновляет нормальную резкость только по уверенным, не замутнённым кадрам."""

        sharpness = self._measure_frame_sharpness(frame)
        if self._sharpness_baseline is None:
            self._sharpness_baseline = max(sharpness, 1e-6)
            return

        alpha = 0.06
        self._sharpness_baseline = self._sharpness_baseline * (1.0 - alpha) + sharpness * alpha

    def _blur_hold_active(self) -> bool:
        """Возвращает, что мы всё ещё восстанавливаемся после замутнения."""

        return self.config.blur_hold_enabled and (self._degraded_frames > 0 or self._blur_hold_frames > 0)

    def _current_max_lost_frames(self) -> int:
        """Даёт дополнительный бюджет потери, если кадр недавно был замутнён."""

        if not self._blur_hold_active():
            return self.config.max_lost_frames
        return self.config.max_lost_frames + self.config.blur_hold_max_frames

    def _detect_edge_contact(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> set[str]:
        """Возвращает стороны кадра, к которым цель подошла вплотную."""

        frame_h, frame_w = frame_shape[:2]
        margin = max(0, int(self.config.edge_exit_margin))
        edges: set[str] = set()
        if bbox.x <= margin:
            edges.add("left")
        if bbox.y <= margin:
            edges.add("top")
        if bbox.x2 >= frame_w - margin:
            edges.add("right")
        if bbox.y2 >= frame_h - margin:
            edges.add("bottom")
        return edges

    def _update_exit_edges(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        """Запоминает, что цель могла уйти за границу кадра."""

        self._exit_edges = self._detect_edge_contact(bbox, frame_shape)

    def _is_invalid_edge_candidate(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        """Не даёт после выхода за край перескочить на похожее пятно внутри кадра."""

        if not self._exit_edges:
            return False
        return not self._exit_edges.intersection(self._detect_edge_contact(bbox, frame_shape))

    def _is_invalid_motion_candidate(self, bbox: BoundingBox, predicted_bbox: BoundingBox) -> bool:
        """Отсекает кандидата, который ломает уже набранную траекторию."""

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
            if self._blur_hold_active():
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
        """Разрешает обновлять траекторию только по достаточно надёжному совпадению."""

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
                if self._blur_hold_active():
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

    def _update_camera_offset(self, global_motion: FrameStabilizerResult) -> None:
        """Копит сдвиг камеры с момента выбора текущей цели."""

        if not global_motion.valid:
            return
        self._camera_offset += np.array([global_motion.dx, global_motion.dy], dtype=np.float32)

    def _to_motion_model_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """Переводит bbox из координат кадра в координаты, компенсированные сдвигом камеры."""

        center = np.array(bbox.center, dtype=np.float32) - self._camera_offset
        return BoundingBox.from_center(center[0], center[1], bbox.width, bbox.height)

    def _from_motion_model_bbox(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Возвращает прогноз модели движения обратно в координаты текущего кадра."""

        assert self._bbox is not None
        center = np.array(bbox.center, dtype=np.float32) + self._camera_offset
        return BoundingBox.from_center(center[0], center[1], self._bbox.width, self._bbox.height).clamp(frame_shape)

    def _update_velocity(self, measured_bbox: BoundingBox, global_motion: FrameStabilizerResult) -> None:
        """Обновляет остаточную скорость цели относительно движения камеры."""
        assert self._bbox is not None
        previous_center = np.array(self._bbox.center, dtype=np.float32)
        motion_shift = np.array([global_motion.dx, global_motion.dy], dtype=np.float32) if global_motion.valid else 0.0
        baseline_center = previous_center + motion_shift
        measured_center = np.array(measured_bbox.center, dtype=np.float32)
        new_velocity = measured_center - baseline_center
        alpha = self.config.velocity_alpha
        self._residual_velocity = self._residual_velocity * (1.0 - alpha) + new_velocity * alpha

    def _update_templates(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Аккуратно обновляет адаптивный шаблон, когда мы достаточно уверены в цели."""
        assert self._canonical_size is not None
        assert self._adaptive_gray is not None and self._adaptive_grad is not None

        gray_patch = _crop(frame.normalized, bbox)
        grad_patch = _crop(frame.gradient, bbox)
        if gray_patch is None or grad_patch is None:
            return

        new_gray = _safe_resize(gray_patch, self._canonical_size).astype(np.float32)
        new_grad = _safe_resize(grad_patch, self._canonical_size).astype(np.float32)
        alpha = self.config.template_alpha
        adaptive_gray = self._adaptive_gray.astype(np.float32)
        adaptive_grad = self._adaptive_grad.astype(np.float32)
        self._adaptive_gray = cv2.addWeighted(new_gray, alpha, adaptive_gray, 1.0 - alpha, 0.0).astype(np.uint8)
        self._adaptive_grad = cv2.addWeighted(new_grad, alpha, adaptive_grad, 1.0 - alpha, 0.0).astype(np.uint8)

    def _can_update_templates(self, bbox: BoundingBox, score: float) -> bool:
        """Не даёт адаптивному шаблону обучиться на соседней похожей цели."""

        if score < self.config.template_update_threshold:
            return False
        if self._canonical_size is None:
            return True

        canonical_w, canonical_h = self._canonical_size
        width_ratio = bbox.width / max(float(canonical_w), 1.0)
        height_ratio = bbox.height / max(float(canonical_h), 1.0)
        if max(width_ratio, height_ratio) > 1.28 and score < self.config.template_update_threshold + 0.18:
            return False
        return True

    def _stabilize_bbox_size(
        self,
        measured_bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Не даёт рамке прыгать по размеру слишком резко."""
        if self._bbox is None:
            return measured_bbox

        if self._lost_frames == 0:
            min_scale = self.config.max_size_shrink
            max_scale = self.config.max_size_growth
        else:
            min_scale = self.config.max_size_shrink_on_reacquire
            max_scale = self.config.max_size_growth_on_reacquire

        min_width = max(self.config.min_box_size, int(round(self._bbox.width * min_scale)))
        max_width = max(min_width, int(round(self._bbox.width * max_scale)))
        min_height = max(self.config.min_box_size, int(round(self._bbox.height * min_scale)))
        max_height = max(min_height, int(round(self._bbox.height * max_scale)))

        stabilized_width = min(max(measured_bbox.width, min_width), max_width)
        stabilized_height = min(max(measured_bbox.height, min_height), max_height)
        if self._canonical_size is not None:
            initial_w, initial_h = self._canonical_size
            initial_growth = self._allowed_growth_from_initial(initial_w, initial_h)
            max_initial_width = max(self.config.min_box_size, int(round(initial_w * initial_growth)))
            max_initial_height = max(self.config.min_box_size, int(round(initial_h * initial_growth)))
            stabilized_width = min(stabilized_width, max_initial_width)
            stabilized_height = min(stabilized_height, max_initial_height)
        cx, cy = measured_bbox.center
        return BoundingBox.from_center(cx, cy, stabilized_width, stabilized_height).clamp(frame_shape)

    def _allowed_growth_from_initial(self, initial_width: int, initial_height: int) -> float:
        """Даёт крупным целям расти, но не позволяет маленькому клику поглотить соседей."""

        configured_growth = max(1.0, float(self.config.max_size_growth_from_initial))
        initial_max_side = max(initial_width, initial_height)
        if initial_max_side < 24:
            return min(configured_growth, 2.0)
        if initial_max_side < 40:
            return min(configured_growth, 1.45)
        if initial_max_side < 70:
            return min(configured_growth, 1.2)
        return configured_growth

    def _initialize_feature_points(self, frame: ProcessedFrame, bbox: BoundingBox, force: bool = False) -> None:
        """Набирает хорошие точки внутри объекта для локального сопровождения."""
        if not force and self._tracked_points is not None and len(self._tracked_points) >= self.config.min_feature_points:
            return

        mask = np.zeros_like(frame.normalized, dtype=np.uint8)
        clamped = bbox.clamp(frame.bgr.shape)
        mask[clamped.y:clamped.y2, clamped.x:clamped.x2] = 255

        points = cv2.goodFeaturesToTrack(
            frame.gradient,
            maxCorners=self.config.max_feature_points,
            qualityLevel=self.config.feature_quality_level,
            minDistance=self.config.feature_min_distance,
            mask=mask,
            blockSize=3,
        )
        if points is None:
            points = cv2.goodFeaturesToTrack(
                frame.normalized,
                maxCorners=self.config.max_feature_points,
                qualityLevel=self.config.feature_quality_level,
                minDistance=self.config.feature_min_distance,
                mask=mask,
                blockSize=3,
            )

        if points is None or len(points) < self.config.min_feature_points:
            self._tracked_points = None
            return

        self._tracked_points = points.astype(np.float32)
        self._update_point_offset(clamped)
        self._frames_since_feature_refresh = 0

    def _refresh_feature_points(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Освежает набор точек, когда старые начинают заканчиваться."""
        usable_points = self._filter_points_inside_bbox(bbox)
        if usable_points is not None:
            self._tracked_points = usable_points
            self._update_point_offset(bbox)

        self._frames_since_feature_refresh += 1
        should_refresh = (
            self._tracked_points is None
            or len(self._tracked_points) < self.config.min_feature_points
            or self._frames_since_feature_refresh >= self.config.feature_refresh_interval
        )

        if should_refresh:
            self._initialize_feature_points(frame, bbox, force=True)

    def _filter_points_inside_bbox(self, bbox: BoundingBox) -> np.ndarray | None:
        """Оставляет только те точки, которые всё ещё лежат рядом с текущей целью."""
        if self._tracked_points is None:
            return None

        x1 = bbox.x - 3
        y1 = bbox.y - 3
        x2 = bbox.x2 + 3
        y2 = bbox.y2 + 3
        points = self._tracked_points.reshape(-1, 2)
        inside_mask = (
            (points[:, 0] >= x1)
            & (points[:, 0] <= x2)
            & (points[:, 1] >= y1)
            & (points[:, 1] <= y2)
        )
        filtered = points[inside_mask]
        if len(filtered) < self.config.min_feature_points:
            return None
        return filtered.reshape(-1, 1, 2).astype(np.float32)

    def _update_point_offset(self, bbox: BoundingBox) -> None:
        """Запоминает, где центр объекта находится относительно набора точек."""
        if self._tracked_points is None or len(self._tracked_points) == 0:
            self._point_center_offset[:] = 0.0
            return
        point_center = np.median(self._tracked_points.reshape(-1, 2), axis=0)
        bbox_center = np.array(bbox.center, dtype=np.float32)
        self._point_center_offset = bbox_center - point_center

    def resume_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        track_id: int,
    ) -> TrackSnapshot:
        """Возобновляет сопровождение по подтверждённому recovery-bbox без смены track_id.

        Используется из pipeline после того, как track-before-confirm
        подтвердил кандидата от recoverer. Трек продолжается под тем же
        ID, что и до потери; ``_next_track_id`` не инкрементируется.
        """

        clamped = bbox.clamp(frame.bgr.shape)
        canonical_w = max(self.config.min_box_size, clamped.width)
        canonical_h = max(self.config.min_box_size, clamped.height)
        canonical_size = (canonical_w, canonical_h)

        gray_patch = _crop(frame.normalized, clamped)
        grad_patch = _crop(frame.gradient, clamped)
        if gray_patch is None or grad_patch is None:
            return self.reset()

        self._track_id = track_id
        self._canonical_size = canonical_size
        self._bbox = clamped
        self._predicted_bbox = clamped
        self._search_region = clamped.pad(
            self.config.search_margin, self.config.search_margin
        ).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._state = TrackerState.TRACKING
        self._score = 1.0
        self._message = f"Resumed target #{track_id}"
        self._residual_velocity[:] = 0.0
        self._camera_offset[:] = 0.0
        self._target_polarity = self._resolve_target_polarity(frame, clamped, "unknown")
        self._motion_model.initialize(self._to_motion_model_bbox(clamped))
        self._sharpness_baseline = self._measure_frame_sharpness(frame)
        self._degraded_frames = 0
        self._blur_hold_frames = 0

        self._long_term_gray = _safe_resize(gray_patch, canonical_size)
        self._long_term_grad = _safe_resize(grad_patch, canonical_size)
        self._adaptive_gray = self._long_term_gray.copy()
        self._adaptive_grad = self._long_term_grad.copy()

        self._initialize_feature_points(frame, clamped, force=True)
        self._previous_normalized = frame.normalized.copy()
        self._update_exit_edges(clamped, frame.bgr.shape)
        return self.snapshot(FrameStabilizerResult())
