from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Self

import cv2
import numpy as np

from ....config import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame, TrackerState
from ..result import TargetTrackingResult
from ...frame_stabilization import FrameStabilizerResult
from ...target_selection import TargetPolarity
from .base_target_tracker import BaseTargetTracker
from ..type import TargetTrackerType


@dataclass(frozen=True, slots=True)
class IrstContrastTargetTrackerConfig:
    """Хранит настройки IRST-трекера по локальному контрасту."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetTrackerType] = TargetTrackerType.IRST_CONTRAST

    # Радиус поиска цели вокруг клика при старте трека.
    click_search_radius: int = 32
    # Размер fallback-bbox, если по клику не найден контрастный blob.
    click_fallback_size: int = 16

    # Размер ядра для оценки локального объекта.
    filter_kernel: int = 3
    # Размер ядра для оценки локального фона.
    background_kernel: int = 15
    # Минимальное значение карты контраста для попадания в blob-маску.
    contrast_threshold: float = 18.0
    # Минимальная площадь blob-кандидата.
    min_blob_area: int = 4
    # Максимальная площадь blob-кандидата.
    max_blob_area: int = 900
    # Включает морфологическое объединение близких контрастных пикселей.
    dilate_candidates: bool = True
    # Размер ядра объединения близких контрастных пикселей.
    candidate_dilate_kernel: int = 3
    # Количество итераций объединения близких контрастных пикселей.
    candidate_dilate_iterations: int = 1

    # Минимальный радиус search gate вокруг прогноза.
    min_gate: int = 24
    # Максимальный радиус search gate вокруг прогноза.
    max_gate: int = 96
    # Увеличение search gate за каждый потерянный кадр.
    gate_growth: int = 8
    # Максимальная физически допустимая скорость цели в пикселях за кадр.
    max_speed_px_per_frame: float = 24.0
    # Множитель запаса для физического ограничения скорости.
    physical_gate_lost_frame_multiplier: float = 1.5

    # Максимальное количество кадров потери перед сбросом трека.
    max_lost_frames: int = 15

    # Шум позиции в Kalman-фильтре.
    kalman_process_noise_pos: float = 1e-2
    # Шум скорости в Kalman-фильтре.
    kalman_process_noise_vel: float = 1e-1
    # Шум измерения позиции в Kalman-фильтре.
    kalman_measurement_noise: float = 1e-1
    # Начальная ковариация ошибки Kalman-фильтра.
    kalman_initial_error_cov: float = 4.0

    # Включает удержание трека при замутнении кадра.
    blur_hold_enabled: bool = True
    # Доля падения резкости, после которой кадр считается деградированным.
    blur_sharpness_drop_ratio: float = 0.45
    # Верхний перцентиль Лапласиана для оценки резкости.
    sharpness_percentile: float = 90.0
    # Верхняя граница центральной ROI по высоте для оценки резкости.
    sharpness_roi_y_min: float = 0.15
    # Нижняя граница центральной ROI по высоте для оценки резкости.
    sharpness_roi_y_max: float = 0.85
    # Левая граница центральной ROI по ширине для оценки резкости.
    sharpness_roi_x_min: float = 0.08
    # Правая граница центральной ROI по ширине для оценки резкости.
    sharpness_roi_x_max: float = 0.92
    # Размер ядра Лапласиана для оценки резкости.
    sharpness_laplacian_kernel: int = 3
    # Скорость адаптации базовой резкости по хорошим кадрам.
    sharpness_baseline_alpha: float = 0.05
    # Минимальное значение baseline, чтобы не делить на ноль.
    min_sharpness_baseline: float = 1e-6

    def __post_init__(self) -> None:
        """Проверить корректность параметров IRST-трекера."""
        self._validate_positive_int(self.click_search_radius, "click_search_radius")
        self._validate_positive_int(self.click_fallback_size, "click_fallback_size")
        self._validate_odd_positive_kernel(self.filter_kernel, "filter_kernel")
        self._validate_odd_positive_kernel(self.background_kernel, "background_kernel")
        self._validate_non_negative_float(self.contrast_threshold, "contrast_threshold")
        self._validate_positive_int(self.min_blob_area, "min_blob_area")
        self._validate_positive_int(self.max_blob_area, "max_blob_area")

        if self.min_blob_area > self.max_blob_area:
            raise ValueError("min_blob_area must be less than or equal to max_blob_area.")

        self._validate_odd_positive_kernel(
            self.candidate_dilate_kernel,
            "candidate_dilate_kernel",
        )
        self._validate_non_negative_int(
            self.candidate_dilate_iterations,
            "candidate_dilate_iterations",
        )

        self._validate_positive_int(self.min_gate, "min_gate")
        self._validate_positive_int(self.max_gate, "max_gate")
        self._validate_non_negative_int(self.gate_growth, "gate_growth")
        self._validate_positive_float(self.max_speed_px_per_frame, "max_speed_px_per_frame")
        self._validate_positive_float(
            self.physical_gate_lost_frame_multiplier,
            "physical_gate_lost_frame_multiplier",
        )

        if self.min_gate > self.max_gate:
            raise ValueError("min_gate must be less than or equal to max_gate.")

        self._validate_non_negative_int(self.max_lost_frames, "max_lost_frames")
        self._validate_positive_float(self.kalman_process_noise_pos, "kalman_process_noise_pos")
        self._validate_positive_float(self.kalman_process_noise_vel, "kalman_process_noise_vel")
        self._validate_positive_float(self.kalman_measurement_noise, "kalman_measurement_noise")
        self._validate_positive_float(self.kalman_initial_error_cov, "kalman_initial_error_cov")
        self._validate_ratio(self.blur_sharpness_drop_ratio, "blur_sharpness_drop_ratio")
        self._validate_percent(self.sharpness_percentile, "sharpness_percentile")
        self._validate_ratio(self.sharpness_roi_y_min, "sharpness_roi_y_min")
        self._validate_ratio(self.sharpness_roi_y_max, "sharpness_roi_y_max")
        self._validate_ratio(self.sharpness_roi_x_min, "sharpness_roi_x_min")
        self._validate_ratio(self.sharpness_roi_x_max, "sharpness_roi_x_max")
        self._validate_sobel_like_kernel(
            self.sharpness_laplacian_kernel,
            "sharpness_laplacian_kernel",
        )
        self._validate_ratio(self.sharpness_baseline_alpha, "sharpness_baseline_alpha")
        self._validate_positive_float(self.min_sharpness_baseline, "min_sharpness_baseline")

        if self.sharpness_roi_y_min >= self.sharpness_roi_y_max:
            raise ValueError("sharpness_roi_y_min must be less than sharpness_roi_y_max.")
        if self.sharpness_roi_x_min >= self.sharpness_roi_x_max:
            raise ValueError("sharpness_roi_x_min must be less than sharpness_roi_x_max.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_bool_to(kwargs, "dilate_candidates")
        reader.pop_bool_to(kwargs, "blur_hold_enabled")

        for field_name in (
            "click_search_radius",
            "click_fallback_size",
            "filter_kernel",
            "background_kernel",
            "min_blob_area",
            "max_blob_area",
            "candidate_dilate_kernel",
            "candidate_dilate_iterations",
            "min_gate",
            "max_gate",
            "gate_growth",
            "max_lost_frames",
            "sharpness_laplacian_kernel",
        ):
            reader.pop_int_to(kwargs, field_name)

        for field_name in (
            "contrast_threshold",
            "max_speed_px_per_frame",
            "physical_gate_lost_frame_multiplier",
            "kalman_process_noise_pos",
            "kalman_process_noise_vel",
            "kalman_measurement_noise",
            "kalman_initial_error_cov",
            "blur_sharpness_drop_ratio",
            "sharpness_percentile",
            "sharpness_roi_y_min",
            "sharpness_roi_y_max",
            "sharpness_roi_x_min",
            "sharpness_roi_x_max",
            "sharpness_baseline_alpha",
            "min_sharpness_baseline",
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

    @staticmethod
    def _validate_ratio(value: float, field_name: str) -> None:
        """Проверить, что значение находится в диапазоне (0, 1]."""
        if not 0 < value <= 1:
            raise ValueError(f"{field_name} must be in range (0, 1].")

    @staticmethod
    def _validate_percent(value: float, field_name: str) -> None:
        """Проверить, что значение является процентом."""
        if not 0 <= value <= 100:
            raise ValueError(f"{field_name} must be in range [0, 100].")

    @staticmethod
    def _validate_odd_positive_kernel(value: int, field_name: str) -> None:
        """Проверить, что размер ядра положительный и нечётный."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd.")

    @staticmethod
    def _validate_sobel_like_kernel(value: int, field_name: str) -> None:
        """Проверить размер ядра OpenCV-фильтра."""
        if value not in (1, 3, 5, 7):
            raise ValueError(f"{field_name} must be one of: 1, 3, 5, 7.")


@dataclass(slots=True)
class IrstContrastTargetTracker(BaseTargetTracker):
    """
        Трекер маленькой тепловой цели на базе локального контраста и фильтра Калмана.
        Подход — IRST (Infrared Search and Track / инфракрасный поиск и сопровождение):
        - каждый кадр строится карта локального контраста (яркость пикселя vs его окружение);
        - контрастные кластеры (blob-ы) извлекаются через пороговую фильтрацию и connectedComponents;
        - позиция выбранного blob-а является измерением для фильтра Калмана;
        - Kalman предсказывает следующую позицию, определяет зону захвата (gate);
        - кандидат принимается только если он попадает в gate И физически достижим
          от последней известной позиции за lost_frames кадров при заданной max_speed.
        Не использует:
        - сопоставление шаблонов (template matching);
        - оптический поток по точкам (LK optical flow);
        - phase correlation (ненадёжна при blur + поворот камеры).
    """

    config: IrstContrastTargetTrackerConfig

    _track_id: int | None = field(default=None, init=False)
    _next_track_id: int = field(default=0, init=False)

    _state: TrackerState = field(default=TrackerState.IDLE, init=False)
    _bbox: BoundingBox | None = field(default=None, init=False)
    _predicted_bbox: BoundingBox | None = field(default=None, init=False)
    _search_region: BoundingBox | None = field(default=None, init=False)
    _score: float = field(default=0.0, init=False)
    _lost_frames: int = field(default=0, init=False)
    _message: str = field(default="Click target", init=False)

    _polarity: TargetPolarity = field(default=TargetPolarity.HOT, init=False)
    _canonical_size: tuple[int, int] | None = field(default=None, init=False)

    _kalman: cv2.KalmanFilter = field(init=False, repr=False)
    _kalman_initialized: bool = field(default=False, init=False)

    _sharpness_baseline: float | None = field(default=None, init=False)
    _blur_hold_frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Подготовить внутренний Kalman-фильтр."""
        self._kalman = self._build_kalman()

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TargetTrackingResult:
        """Начать сопровождение цели по клику оператора."""
        search_region = BoundingBox.from_center(
            point[0],
            point[1],
            self.config.click_search_radius * 2,
            self.config.click_search_radius * 2,
        ).clamp(frame.bgr.shape)

        candidates = self._find_candidates(frame=frame, search_region=search_region)

        if candidates:
            best_bbox, best_score = self._select_start_candidate(candidates=candidates, point=point)
        else:
            best_bbox = BoundingBox.from_center(
                point[0],
                point[1],
                self.config.click_fallback_size,
                self.config.click_fallback_size,
            ).clamp(frame.bgr.shape)
            best_score = 0.0

        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._canonical_size = (best_bbox.width, best_bbox.height)
        self._polarity = self._detect_polarity(frame=frame, bbox=best_bbox)
        self._bbox = best_bbox
        self._predicted_bbox = best_bbox
        self._search_region = best_bbox.pad(self.config.min_gate, self.config.min_gate).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._score = best_score
        self._state = TrackerState.TRACKING
        self._message = f"IRST tracking target #{self._track_id}"

        center_x, center_y = best_bbox.center
        self._init_kalman(center_x, center_y)

        self._sharpness_baseline = self._measure_sharpness(frame)
        self._blur_hold_frames = 0

        return self.snapshot(FrameStabilizerResult())

    def update(self, frame: ProcessedFrame, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Обновить трек на новом кадре."""
        if self._state == TrackerState.IDLE or not self._kalman_initialized:
            self._message = "Click target"
            return self.snapshot(motion)

        if self._update_blur_state(frame):
            return self._coast_on_degraded_frame(frame=frame, motion=motion)

        predicted_center = self._kalman_predict()
        predicted_bbox = self._build_predicted_bbox(frame_shape=frame.bgr.shape, predicted_center=predicted_center)
        self._predicted_bbox = predicted_bbox
        self._search_region = self._build_search_region(frame_shape=frame.bgr.shape, predicted_bbox=predicted_bbox)

        candidates = self._find_candidates(frame=frame, search_region=self._search_region)
        best = self._associate(
            candidates=candidates,
            predicted_center=predicted_center,
            last_bbox=self._bbox,
            lost_frames=self._lost_frames,
        )

        if best is not None:
            return self._accept_candidate(
                frame=frame,
                motion=motion,
                predicted_bbox=predicted_bbox,
                candidate=best,
            )

        return self._mark_lost(motion)

    def reset(self) -> TargetTrackingResult:
        """Сбросить текущее состояние трекера."""
        self._track_id = None
        self._state = TrackerState.IDLE
        self._bbox = None
        self._predicted_bbox = None
        self._search_region = None
        self._score = 0.0
        self._lost_frames = 0
        self._message = "Tracker reset"
        self._canonical_size = None
        self._kalman_initialized = False
        self._sharpness_baseline = None
        self._blur_hold_frames = 0

        return self.snapshot(FrameStabilizerResult())

    def snapshot(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Вернуть текущее состояние трека."""
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

    def resume_tracking(self, frame: ProcessedFrame, bbox: BoundingBox, track_id: int) -> TargetTrackingResult:
        """Возобновить сопровождение цели с заданными bbox и track_id."""
        clamped_bbox = bbox.clamp(frame.bgr.shape)
        center_x, center_y = clamped_bbox.center

        self._track_id = track_id
        self._next_track_id = max(self._next_track_id, track_id + 1)
        self._bbox = clamped_bbox
        self._predicted_bbox = clamped_bbox
        self._search_region = clamped_bbox.pad(
            self.config.min_gate,
            self.config.min_gate,
        ).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._score = 1.0
        self._state = TrackerState.TRACKING
        self._message = f"IRST resumed target #{track_id}"
        self._polarity = self._detect_polarity(frame=frame, bbox=clamped_bbox)

        if self._kalman_initialized:
            state = self._kalman.statePost.copy()
            state[0, 0] = center_x
            state[1, 0] = center_y
            self._kalman.statePost = state
            self._kalman.statePre = state.copy()
        else:
            self._init_kalman(center_x, center_y)

        self._sharpness_baseline = self._measure_sharpness(frame)
        self._blur_hold_frames = 0

        return self.snapshot(FrameStabilizerResult())

    def _compute_contrast_map(self, gray: np.ndarray) -> np.ndarray:
        """Вычислить карту локального контраста тепловой цели."""
        gray_float = gray.astype(np.float32)

        inner_kernel = np.ones((self.config.filter_kernel, self.config.filter_kernel), dtype=np.uint8)
        local_max = cv2.dilate(gray, inner_kernel).astype(np.float32)
        local_min = cv2.erode(gray, inner_kernel).astype(np.float32)
        background = cv2.blur(gray_float, (self.config.background_kernel, self.config.background_kernel))

        hot_contrast = local_max - background
        cold_contrast = background - local_min
        contrast_map = np.maximum(hot_contrast, cold_contrast)

        return np.clip(contrast_map, 0.0, 255.0)

    def _find_candidates(self, frame: ProcessedFrame, search_region: BoundingBox | None) -> list[tuple[BoundingBox, float]]:
        """Найти контрастные blob-кандидаты в зоне поиска."""
        gray = frame.gray
        contrast_map = self._compute_contrast_map(gray)

        if search_region is not None:
            region = search_region.clamp(gray.shape)
            mask = np.zeros(contrast_map.shape, dtype=np.float32)
            mask[region.y:region.y2, region.x:region.x2] = 1.0
            contrast_map = contrast_map * mask

        binary = (contrast_map >= self.config.contrast_threshold).astype(np.uint8) * 255

        if self.config.dilate_candidates and self.config.candidate_dilate_iterations > 0:
            kernel = np.ones((self.config.candidate_dilate_kernel, self.config.candidate_dilate_kernel), dtype=np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=self.config.candidate_dilate_iterations)

        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        candidates: list[tuple[BoundingBox, float]] = []

        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])

            if not self.config.min_blob_area <= area <= self.config.max_blob_area:
                continue

            center_x = float(centroids[label][0])
            center_y = float(centroids[label][1])
            blob_width = int(stats[label, cv2.CC_STAT_WIDTH])
            blob_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_width, bbox_height = self._resolve_candidate_size(blob_width=blob_width, blob_height=blob_height)

            bbox = BoundingBox.from_center(center_x, center_y, bbox_width, bbox_height).clamp(gray.shape)

            blob_pixels = contrast_map[labels == label]
            score = float(np.mean(blob_pixels)) / 255.0 if blob_pixels.size > 0 else 0.0

            candidates.append((bbox, score))

        candidates.sort(key=lambda candidate: -candidate[1])
        return candidates

    def _associate(
        self,
        candidates: list[tuple[BoundingBox, float]],
        predicted_center: tuple[float, float],
        last_bbox: BoundingBox | None,
        lost_frames: int,
    ) -> tuple[BoundingBox, float] | None:
        """Выбрать лучшего кандидата с учётом Kalman gate и физического gate."""
        if not candidates:
            return None

        gate_radius = self._current_gate_radius(lost_frames)

        if last_bbox is not None:
            max_physical_distance = (
                self.config.max_speed_px_per_frame
                * max(lost_frames, 1)
                * self.config.physical_gate_lost_frame_multiplier
            )
            last_center_x, last_center_y = last_bbox.center
        else:
            max_physical_distance = float("inf")
            last_center_x, last_center_y = predicted_center

        predicted_x, predicted_y = predicted_center
        best: tuple[BoundingBox, float] | None = None
        best_distance = float("inf")

        for bbox, score in candidates:
            center_x, center_y = bbox.center
            distance_from_prediction = math.hypot(center_x - predicted_x, center_y - predicted_y)
            distance_from_last = math.hypot(center_x - last_center_x, center_y - last_center_y)

            if distance_from_prediction > gate_radius:
                continue
            if distance_from_last > max_physical_distance:
                continue

            if distance_from_prediction < best_distance:
                best_distance = distance_from_prediction
                best = (bbox, score)

        return best

    def _build_kalman(self) -> cv2.KalmanFilter:
        """Создать Kalman-фильтр для центра цели."""
        kalman = cv2.KalmanFilter(4, 2)

        kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        return kalman

    def _init_kalman(self, center_x: float, center_y: float) -> None:
        """Инициализировать Kalman-фильтр известной позицией цели."""
        self._kalman.processNoiseCov = np.diag(
            [
                self.config.kalman_process_noise_pos,
                self.config.kalman_process_noise_pos,
                self.config.kalman_process_noise_vel,
                self.config.kalman_process_noise_vel,
            ]
        ).astype(np.float32)
        self._kalman.measurementNoiseCov = np.diag([
                self.config.kalman_measurement_noise,
                self.config.kalman_measurement_noise,
            ]
        ).astype(np.float32)
        self._kalman.errorCovPost = (np.eye(4, dtype=np.float32) * self.config.kalman_initial_error_cov)
        self._kalman.statePost = np.array([[center_x], [center_y], [0.0], [0.0]], dtype=np.float32)
        self._kalman.statePre = self._kalman.statePost.copy()
        self._kalman_initialized = True

    def _kalman_predict(self) -> tuple[float, float]:
        """Сделать шаг прогноза и вернуть ожидаемый центр цели."""
        predicted = self._kalman.predict()
        return float(predicted[0, 0]), float(predicted[1, 0])

    def _kalman_update(self, center_x: float, center_y: float) -> None:
        """Обновить Kalman-фильтр измеренным центром цели."""
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        self._kalman.correct(measurement)

    def _detect_polarity(self, frame: ProcessedFrame, bbox: BoundingBox) -> TargetPolarity:
        """Определить, цель горячее или холоднее локального фона."""
        gray = frame.gray
        clamped = bbox.clamp(gray.shape)
        object_patch = gray[clamped.y:clamped.y2, clamped.x:clamped.x2]

        if object_patch.size == 0:
            return TargetPolarity.HOT

        margin = max(4, min(clamped.width, clamped.height))
        outer = clamped.pad(margin, margin).clamp(gray.shape)
        outer_patch = gray[outer.y:outer.y2, outer.x:outer.x2]

        ring_mask = np.ones(outer_patch.shape, dtype=bool)
        inner_y1 = clamped.y - outer.y
        inner_x1 = clamped.x - outer.x
        inner_y2 = inner_y1 + clamped.height
        inner_x2 = inner_x1 + clamped.width
        ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        background_values = outer_patch[ring_mask]

        if background_values.size == 0:
            return TargetPolarity.HOT

        background_mean = float(np.mean(background_values))
        object_mean = float(np.mean(object_patch))

        if object_mean >= background_mean:
            return TargetPolarity.HOT

        return TargetPolarity.COLD

    def _measure_sharpness(self, frame: ProcessedFrame) -> float:
        """Оценить резкость центральной части кадра через Лапласиан."""
        gray = frame.gray
        height, width = gray.shape[:2]

        roi = gray[
            int(height * self.config.sharpness_roi_y_min): int(height * self.config.sharpness_roi_y_max),
            int(width * self.config.sharpness_roi_x_min): int(width * self.config.sharpness_roi_x_max),
        ]

        if roi.size == 0:
            roi = gray

        laplacian = cv2.Laplacian(roi, cv2.CV_32F, ksize=self.config.sharpness_laplacian_kernel)

        return float(np.percentile(np.abs(laplacian), self.config.sharpness_percentile))

    def _update_blur_state(self, frame: ProcessedFrame) -> bool:
        """Проверить, деградировал ли текущий кадр по резкости."""
        if not self.config.blur_hold_enabled:
            self._blur_hold_frames = 0
            return False

        sharpness = self._measure_sharpness(frame)

        if self._sharpness_baseline is None:
            self._sharpness_baseline = max(sharpness, self.config.min_sharpness_baseline)
            return False

        baseline = max(self._sharpness_baseline, self.config.min_sharpness_baseline)
        degraded = sharpness <= baseline * self.config.blur_sharpness_drop_ratio

        if degraded:
            self._blur_hold_frames += 1
        else:
            self._blur_hold_frames = max(0, self._blur_hold_frames - 1)

        return degraded

    def _update_sharpness_baseline(self, frame: ProcessedFrame) -> None:
        """Медленно адаптировать базовую резкость по хорошим кадрам."""
        sharpness = self._measure_sharpness(frame)

        if self._sharpness_baseline is None:
            self._sharpness_baseline = max(sharpness, self.config.min_sharpness_baseline)
            return

        alpha = self.config.sharpness_baseline_alpha
        self._sharpness_baseline = self._sharpness_baseline * (1 - alpha) + sharpness * alpha

    def _select_start_candidate(self, candidates: list[tuple[BoundingBox, float]], point: tuple[int, int]) -> tuple[BoundingBox, float]:
        """Выбрать стартовый blob около клика с учётом score и расстояния."""
        reference_distance = max(self.config.click_search_radius / 4.0, 1.0)

        return max(
            candidates,
            key=lambda candidate: candidate[1]
            / (
                1.0 + math.hypot(candidate[0].center[0] - point[0], candidate[0].center[1] - point[1])
                / reference_distance
            ),
        )

    def _build_predicted_bbox(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        predicted_center: tuple[float, float],
    ) -> BoundingBox:
        """Построить bbox прогноза по предсказанному центру."""
        width = self._canonical_size[0] if self._canonical_size else self.config.click_fallback_size
        height = self._canonical_size[1] if self._canonical_size else self.config.click_fallback_size

        return BoundingBox.from_center(predicted_center[0], predicted_center[1], width, height).clamp(frame_shape)

    def _build_search_region(self, frame_shape: tuple[int, int] | tuple[int, int, int], predicted_bbox: BoundingBox) -> BoundingBox:
        """Построить область поиска вокруг прогнозного bbox."""
        gate_radius = self._current_gate_radius(self._lost_frames)
        return predicted_bbox.pad(gate_radius, gate_radius).clamp(frame_shape)

    def _accept_candidate(
        self,
        frame: ProcessedFrame,
        motion: FrameStabilizerResult,
        predicted_bbox: BoundingBox,
        candidate: tuple[BoundingBox, float],
    ) -> TargetTrackingResult:
        """Принять найденного кандидата как текущее положение цели."""
        bbox, score = candidate
        center_x, center_y = bbox.center
        self._kalman_update(center_x, center_y)

        self._bbox = bbox
        self._predicted_bbox = predicted_bbox
        self._score = score
        self._lost_frames = 0
        self._state = TrackerState.TRACKING
        self._message = f"IRST tracking target #{self._track_id}"
        self._update_sharpness_baseline(frame)

        return self.snapshot(motion)

    def _coast_on_degraded_frame(self, frame: ProcessedFrame, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Продолжить прогноз без детекции на замутнённом кадре."""
        predicted_center = self._kalman_predict()
        self._predicted_bbox = self._build_predicted_bbox(
            frame_shape=frame.bgr.shape,
            predicted_center=predicted_center,
        )
        self._search_region = self._build_search_region(
            frame_shape=frame.bgr.shape,
            predicted_bbox=self._predicted_bbox,
        )
        self._lost_frames += 1
        self._score = 0.0
        self._state = TrackerState.SEARCHING
        self._message = f"Frame blurred, coasting #{self._track_id}"

        if self._lost_frames > self.config.max_lost_frames:
            return self._go_idle()

        return self.snapshot(motion)

    def _mark_lost(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Отметить кадр как потерю цели."""
        self._lost_frames += 1
        self._score = 0.0
        self._state = TrackerState.SEARCHING
        self._message = f"Searching for target #{self._track_id}"

        if self._lost_frames > self.config.max_lost_frames:
            return self._go_idle()

        return self.snapshot(motion)

    def _go_idle(self) -> TargetTrackingResult:
        """Перевести трекер в IDLE после исчерпания лимита потери."""
        self._state = TrackerState.IDLE
        self._bbox = None
        self._message = f"Target #{self._track_id} lost, select target again"
        self._track_id = None
        self._kalman_initialized = False

        return self.snapshot(FrameStabilizerResult())

    def _current_gate_radius(self, lost_frames: int) -> int:
        """Вернуть текущий радиус search gate."""
        return min(
            self.config.min_gate + lost_frames * self.config.gate_growth,
            self.config.max_gate,
        )

    def _resolve_candidate_size(self, blob_width: int, blob_height: int) -> tuple[int, int]:
        """Вернуть размер bbox-кандидата."""
        if self._canonical_size is not None:
            return self._canonical_size

        return (
            max(blob_width, self.config.click_fallback_size),
            max(blob_height, self.config.click_fallback_size),
        )
