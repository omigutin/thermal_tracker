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

from dataclasses import dataclass

import cv2
import numpy as np

from ...config import ClickSelectionConfig, TrackerConfig
from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame, TrackSnapshot, TrackerState
from ..target_selection import ClickTargetSelector
from .base_target_tracker import BaseSingleTargetTracker


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


@dataclass
class _SearchResult:
    """Лучший найденный кандидат на текущем кадре."""
    bbox: BoundingBox
    score: float
    search_region: BoundingBox


@dataclass
class _PointPrediction:
    """Прогноз положения цели по опорным точкам."""
    bbox: BoundingBox
    confidence: float


class ClickToTrackSingleTargetTracker(BaseSingleTargetTracker):
    """Трекер одной цели по клику.
    Логика простая:
    - после клика выбираем объект вокруг точки;
    - сохраняем два шаблона: долгий и адаптивный;
    - в каждом кадре сначала пробуем понять движение по точкам;
    - затем ищем лучший кандидат по внешнему виду;
    - если цель пропала, расширяем область поиска и пытаемся вернуть тот же ID.
    """

    implementation_name = "hybrid_template_point"
    is_ready = True

    def __init__(self, tracker_config: TrackerConfig, click_config: ClickSelectionConfig) -> None:
        self.config = tracker_config
        self.selector = ClickTargetSelector(click_config)
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
        self._point_center_offset = np.zeros(2, dtype=np.float32)
        self._frames_since_feature_refresh = 0

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
        self._point_center_offset[:] = 0.0
        self._frames_since_feature_refresh = 0
        return self.snapshot(GlobalMotion())

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

        self._long_term_gray = _safe_resize(gray_patch, self._canonical_size)
        self._long_term_grad = _safe_resize(grad_patch, self._canonical_size)
        self._adaptive_gray = self._long_term_gray.copy()
        self._adaptive_grad = self._long_term_grad.copy()

        # Сразу набираем точки на объекте, чтобы не ехать только на шаблоне.
        self._initialize_feature_points(frame, bbox, force=True)
        self._previous_normalized = frame.normalized.copy()
        return self.snapshot(GlobalMotion())

    def update(self, frame: ProcessedFrame, global_motion: GlobalMotion) -> TrackSnapshot:
        """Обновляет трекер на новом кадре."""
        if self._state == TrackerState.IDLE or self._bbox is None:
            self._message = "Click target"
            self._previous_normalized = frame.normalized.copy()
            return self.snapshot(global_motion)

        point_prediction = self._predict_from_points(frame)
        predicted_bbox = point_prediction.bbox if point_prediction is not None else self._predict_bbox(frame.bgr.shape, global_motion)
        search = self._locate_target(frame, predicted_bbox, point_prediction)
        threshold = self.config.track_threshold if self._state == TrackerState.TRACKING else self.config.reacquire_threshold

        if search is not None and search.score >= threshold:
            refined = self.selector.refine(frame, search.bbox)
            measured_bbox = refined.bbox if refined is not None else search.bbox
            measured_bbox = measured_bbox.clamp(frame.bgr.shape)
            measured_bbox = self._stabilize_bbox_size(measured_bbox, frame.bgr.shape)
            self._update_velocity(measured_bbox, global_motion)
            self._bbox = measured_bbox
            self._predicted_bbox = predicted_bbox
            self._search_region = search.search_region
            self._lost_frames = 0
            self._score = search.score
            self._state = TrackerState.TRACKING
            self._message = f"Tracking target #{self._track_id}"

            if search.score >= self.config.template_update_threshold:
                self._update_templates(frame, measured_bbox)

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

        if self._lost_frames > self.config.max_lost_frames:
            self._message = "Target lost, click again"
            self._state = TrackerState.IDLE
            self._bbox = None
            self._tracked_points = None
            self._previous_normalized = frame.normalized.copy()
            return self.snapshot(global_motion)

        self._state = TrackerState.SEARCHING
        self._message = f"Searching for target #{self._track_id}"
        self._previous_normalized = frame.normalized.copy()
        return self.snapshot(global_motion)

    def snapshot(self, global_motion: GlobalMotion) -> TrackSnapshot:
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
        global_motion: GlobalMotion,
    ) -> BoundingBox:
        """Строит прогноз по прошлому боксу, скорости цели и сдвигу камеры."""
        assert self._bbox is not None

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

    def _predict_from_points(self, frame: ProcessedFrame) -> _PointPrediction | None:
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
        return _PointPrediction(bbox=predicted_bbox, confidence=confidence)

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
        point_prediction: _PointPrediction | None,
    ) -> _SearchResult | None:
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

        best_result: _SearchResult | None = None
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
        point_prediction: _PointPrediction | None,
    ) -> _SearchResult | None:
        """Проверяет кандидатов по двум шаблонам: адаптивному и долгому."""
        response_primary = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        response_fallback = cv2.matchTemplate(search_gray, fallback_template, cv2.TM_CCOEFF_NORMED)

        candidates = []
        for response in (response_primary, response_fallback):
            _, max_value, _, max_location = cv2.minMaxLoc(response)
            candidates.append((float(max_value), max_location))

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

        return _SearchResult(bbox=best_bbox, score=best_score, search_region=search_region)

    def _score_candidate(
        self,
        frame: ProcessedFrame,
        candidate_bbox: BoundingBox,
        predicted_bbox: BoundingBox,
        template_score: float,
        search_region: BoundingBox,
        point_prediction: _PointPrediction | None,
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
            + gray_adaptive * 0.30
            + gray_long * 0.18
            + grad_adaptive * 0.16
            + grad_long * 0.08
        )
        return combined - distance_penalty - point_penalty

    def _update_velocity(self, measured_bbox: BoundingBox, global_motion: GlobalMotion) -> None:
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
        cx, cy = measured_bbox.center
        return BoundingBox.from_center(cx, cy, stabilized_width, stabilized_height).clamp(frame_shape)

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
