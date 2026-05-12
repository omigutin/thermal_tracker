"""Трекер маленькой тепловой цели на базе локального контраста и фильтра Калмана.

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

from __future__ import annotations

import math

import cv2
import numpy as np

from ...config import ClickSelectionConfig, IrstTrackerConfig
from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame, TrackSnapshot, TrackerState
from .base_target_tracker import BaseSingleTargetTracker


class IrstSingleTargetTracker(BaseSingleTargetTracker):
    """Трекер одной цели по локальному контрасту с фильтром Калмана.

    Оператор кликает по цели → ближайший контрастный blob захватывается →
    Kalman ведёт трек кадр за кадром.
    """

    def __init__(
        self,
        config: IrstTrackerConfig,
        click_config: ClickSelectionConfig,
    ) -> None:
        self.config = config
        self.click_config = click_config

        # Внутренние идентификаторы
        self._track_id: int | None = None
        self._next_track_id: int = 0

        # Состояние трека
        self._state: TrackerState = TrackerState.IDLE
        self._bbox: BoundingBox | None = None
        self._predicted_bbox: BoundingBox | None = None
        self._search_region: BoundingBox | None = None
        self._score: float = 0.0
        self._lost_frames: int = 0
        self._message: str = "Click target"

        # Полярность цели: "hot" (ярче фона) или "cold" (темнее фона)
        self._polarity: str = "hot"
        # Канонический размер bbox, зафиксированный при старте
        self._canonical_size: tuple[int, int] | None = None

        # Фильтр Калмана: состояние [cx, cy, vx, vy], измерение [cx, cy]
        self._kalman: cv2.KalmanFilter = self._build_kalman()
        self._kalman_initialized: bool = False

        # Базовая резкость кадра для обнаружения blur
        self._sharpness_baseline: float | None = None
        self._blur_hold_frames: int = 0

    # ------------------------------------------------------------------
    # Публичный контракт BaseSingleTargetTracker
    # ------------------------------------------------------------------

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает трек по клику оператора.

        Ищет ближайший контрастный blob в радиусе click_search_radius.
        Если blob не найден — создаёт fallback bbox размером click_fallback_size.
        """
        search_region = BoundingBox.from_center(
            point[0],
            point[1],
            self.config.click_search_radius * 2,
            self.config.click_search_radius * 2,
        ).clamp(frame.bgr.shape)

        candidates = self._find_candidates(frame, search_region)
        if candidates:
            # Выбираем blob по взвешенному критерию: score / (1 + dist/ref).
            # Это даёт высококонтрастным blob-ам приоритет над ближними шумовыми.
            # reference_dist — "нейтральная" дистанция, при которой distance-штраф ~1.
            reference_dist = max(self.config.click_search_radius / 4.0, 1.0)
            best_bbox, best_score = max(
                candidates,
                key=lambda c: c[1] / (
                    1.0 + math.hypot(c[0].center[0] - point[0], c[0].center[1] - point[1]) / reference_dist
                ),
            )
        else:
            # Fallback: bbox фиксированного размера вокруг клика
            s = self.config.click_fallback_size
            best_bbox = BoundingBox.from_center(point[0], point[1], s, s).clamp(frame.bgr.shape)
            best_score = 0.0

        self._track_id = self._next_track_id
        self._next_track_id += 1
        self._canonical_size = (best_bbox.width, best_bbox.height)
        self._polarity = self._detect_polarity(frame, best_bbox)
        self._bbox = best_bbox
        self._predicted_bbox = best_bbox
        self._search_region = best_bbox.pad(self.config.min_gate, self.config.min_gate).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._score = best_score
        self._state = TrackerState.TRACKING
        self._message = f"IRST tracking target #{self._track_id}"

        cx, cy = best_bbox.center
        self._init_kalman(cx, cy)

        self._sharpness_baseline = self._measure_sharpness(frame)
        self._blur_hold_frames = 0

        return self.snapshot(GlobalMotion())

    def update(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        """Обновляет трек на новом кадре."""
        if self._state == TrackerState.IDLE or not self._kalman_initialized:
            self._message = "Click target"
            return self.snapshot(motion)

        # --- Blur detection ---
        frame_degraded = self._update_blur_state(frame)

        # --- Kalman predict ---
        predicted_center = self._kalman_predict()
        frame_h, frame_w = frame.bgr.shape[:2]
        w = self._canonical_size[0] if self._canonical_size else self.config.click_fallback_size
        h = self._canonical_size[1] if self._canonical_size else self.config.click_fallback_size
        predicted_bbox = BoundingBox.from_center(
            predicted_center[0], predicted_center[1], w, h
        ).clamp(frame.bgr.shape)
        self._predicted_bbox = predicted_bbox

        # --- Зона поиска ---
        gate_radius = min(
            self.config.min_gate + self._lost_frames * self.config.gate_growth,
            self.config.max_gate,
        )
        search_region = predicted_bbox.pad(gate_radius, gate_radius).clamp(frame.bgr.shape)
        self._search_region = search_region

        # --- Детекция ---
        if frame_degraded:
            # На замутнённом кадре не детектируем — только коастим
            self._lost_frames += 1
            self._score = 0.0
            self._state = TrackerState.SEARCHING
            self._message = f"Frame blurred, coasting #{self._track_id}"
            if self._lost_frames > self.config.max_lost_frames:
                return self._go_idle()
            return self.snapshot(motion)

        candidates = self._find_candidates(frame, search_region)
        best = self._associate(candidates, predicted_center, self._bbox, self._lost_frames)

        if best is not None:
            bbox, score = best
            self._kalman_update(bbox.center[0], bbox.center[1])
            self._bbox = bbox
            self._predicted_bbox = predicted_bbox
            self._score = score
            self._lost_frames = 0
            self._state = TrackerState.TRACKING
            self._message = f"IRST tracking target #{self._track_id}"
            self._update_sharpness_baseline(frame)
            return self.snapshot(motion)

        # Цель не найдена
        self._lost_frames += 1
        self._score = 0.0
        self._state = TrackerState.SEARCHING
        self._message = f"Searching for target #{self._track_id}"
        if self._lost_frames > self.config.max_lost_frames:
            return self._go_idle()
        return self.snapshot(motion)

    def reset(self) -> TrackSnapshot:
        """Полный сброс трекера."""
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
        return self.snapshot(GlobalMotion())

    def snapshot(self, motion: GlobalMotion) -> TrackSnapshot:
        """Возвращает текущее состояние трека."""
        visible_bbox = self._bbox if self._state == TrackerState.TRACKING else None
        return TrackSnapshot(
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
    ) -> TrackSnapshot:
        """Возобновляет трек с подтверждённым RECOVERING-bbox под прежним track_id."""
        cx, cy = bbox.center
        self._track_id = track_id
        self._bbox = bbox.clamp(frame.bgr.shape)
        self._predicted_bbox = self._bbox
        self._search_region = self._bbox.pad(self.config.min_gate, self.config.min_gate).clamp(frame.bgr.shape)
        self._lost_frames = 0
        self._score = 1.0
        self._state = TrackerState.TRACKING
        self._message = f"IRST resumed target #{track_id}"
        self._polarity = self._detect_polarity(frame, self._bbox)

        # Переинициализируем Kalman: позиция известна, скорость сохраняем из предыдущего состояния
        if self._kalman_initialized:
            state = self._kalman.statePost.copy()
            state[0, 0] = cx
            state[1, 0] = cy
            self._kalman.statePost = state
            self._kalman.statePre = state.copy()
        else:
            self._init_kalman(cx, cy)

        self._sharpness_baseline = self._measure_sharpness(frame)
        self._blur_hold_frames = 0
        return self.snapshot(GlobalMotion())

    # ------------------------------------------------------------------
    # Детектор контраста
    # ------------------------------------------------------------------

    def _compute_contrast_map(self, gray: np.ndarray) -> np.ndarray:
        """Вычисляет карту локального контраста.

        Для каждого пикселя измеряет, насколько его окрестность ярче или темнее
        своего внешнего кольца (фона). Результат — float32-карта, где высокое
        значение означает контрастный объект (цель).
        """
        fk = self.config.filter_kernel
        bk = self.config.background_kernel
        # Гарантируем нечётность ядер для симметричных фильтров
        fk = fk if fk % 2 == 1 else fk + 1
        bk = bk if bk % 2 == 1 else bk + 1

        gray_f = gray.astype(np.float32)

        # Локальный максимум и минимум в узком окне (объект)
        inner_kernel = np.ones((fk, fk), np.uint8)
        local_max = cv2.dilate(gray, inner_kernel).astype(np.float32)
        local_min = cv2.erode(gray, inner_kernel).astype(np.float32)

        # Локальный фон: среднее в широком окне
        background = cv2.blur(gray_f, (bk, bk))

        # Контраст горячей цели: local_max - background
        # Контраст холодной цели: background - local_min
        hot_contrast = local_max - background
        cold_contrast = background - local_min
        contrast_map = np.maximum(hot_contrast, cold_contrast)
        return np.clip(contrast_map, 0.0, 255.0)

    def _find_candidates(
        self,
        frame: ProcessedFrame,
        search_region: BoundingBox | None,
    ) -> list[tuple[BoundingBox, float]]:
        """Находит контрастные blob-кандидаты в заданной зоне поиска.

        Возвращает список (bbox, score), отсортированный по убыванию score.
        """
        gray = frame.gray
        contrast_map = self._compute_contrast_map(gray)

        # Маска зоны поиска
        if search_region is not None:
            mask = np.zeros(contrast_map.shape, dtype=np.float32)
            r = search_region.clamp(gray.shape)
            mask[r.y:r.y2, r.x:r.x2] = 1.0
            contrast_map = contrast_map * mask

        # Пороговая фильтрация
        thresh = self.config.contrast_threshold
        binary = (contrast_map >= thresh).astype(np.uint8) * 255

        # Морфологическое расширение для слияния близких пикселей
        if self.config.filter_kernel > 1:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

        # Connected components — связные компоненты
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        candidates: list[tuple[BoundingBox, float]] = []
        for i in range(1, num_labels):  # 0 = фон
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self.config.min_blob_area <= area <= self.config.max_blob_area):
                continue

            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            blob_w = int(stats[i, cv2.CC_STAT_WIDTH])
            blob_h = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Размер blob в соответствии с канон. размером или дефолтным
            if self._canonical_size is not None:
                cw, ch = self._canonical_size
            else:
                cw = max(blob_w, self.config.click_fallback_size)
                ch = max(blob_h, self.config.click_fallback_size)

            bbox = BoundingBox.from_center(cx, cy, cw, ch).clamp(gray.shape)

            # Score = среднее значение контраста внутри blob
            blob_pixels = contrast_map[labels == i]
            score = float(np.mean(blob_pixels)) / 255.0 if blob_pixels.size > 0 else 0.0

            candidates.append((bbox, score))

        candidates.sort(key=lambda c: -c[1])
        return candidates

    def _associate(
        self,
        candidates: list[tuple[BoundingBox, float]],
        predicted_center: tuple[float, float],
        last_bbox: BoundingBox | None,
        lost_frames: int,
    ) -> tuple[BoundingBox, float] | None:
        """Выбирает лучшего кандидата из списка с учётом физического гейта.

        Двойная проверка:
        1. Kalman gate: расстояние от прогноза ≤ gate_radius
        2. Physical gate: расстояние от last_bbox ≤ max_speed * lost_frames
        """
        if not candidates:
            return None

        gate_radius = min(
            self.config.min_gate + lost_frames * self.config.gate_growth,
            self.config.max_gate,
        )

        # Физически допустимое смещение от последней известной позиции
        if last_bbox is not None:
            max_physical_dist = self.config.max_speed_px_per_frame * max(lost_frames, 1) * 1.5
            last_cx, last_cy = last_bbox.center
        else:
            max_physical_dist = float("inf")
            last_cx, last_cy = predicted_center

        px, py = predicted_center
        best: tuple[BoundingBox, float] | None = None
        best_dist = float("inf")

        for bbox, score in candidates:
            cx, cy = bbox.center
            dist_from_pred = math.hypot(cx - px, cy - py)
            dist_from_last = math.hypot(cx - last_cx, cy - last_cy)

            if dist_from_pred > gate_radius:
                continue
            if dist_from_last > max_physical_dist:
                continue

            if dist_from_pred < best_dist:
                best_dist = dist_from_pred
                best = (bbox, score)

        return best

    # ------------------------------------------------------------------
    # Фильтр Калмана
    # ------------------------------------------------------------------

    def _build_kalman(self) -> cv2.KalmanFilter:
        """Создаёт фильтр Калмана с моделью постоянной скорости.

        Состояние: [cx, cy, vx, vy] — позиция и скорость центроида.
        Измерение: [cx, cy] — наблюдаемый центроид blob.
        """
        kf = cv2.KalmanFilter(4, 2)

        # Матрица перехода: cx' = cx + vx, cy' = cy + vy, vx' = vx, vy' = vy
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Матрица измерений: наблюдаем только cx, cy
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        return kf

    def _init_kalman(self, cx: float, cy: float) -> None:
        """Инициализирует Kalman с известной позицией и нулевой начальной скоростью."""
        kf = self._kalman

        pos_noise = self.config.kalman_process_noise_pos
        vel_noise = self.config.kalman_process_noise_vel
        meas_noise = self.config.kalman_measurement_noise

        # Шум процесса
        kf.processNoiseCov = np.diag(
            [pos_noise, pos_noise, vel_noise, vel_noise]
        ).astype(np.float32)

        # Шум измерений
        kf.measurementNoiseCov = np.diag(
            [meas_noise, meas_noise]
        ).astype(np.float32)

        # Начальная ковариация ошибки
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 4.0

        # Начальное состояние
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        kf.statePre = kf.statePost.copy()

        self._kalman_initialized = True

    def _kalman_predict(self) -> tuple[float, float]:
        """Делает шаг предсказания и возвращает ожидаемый центр цели."""
        predicted = self._kalman.predict()
        return float(predicted[0, 0]), float(predicted[1, 0])

    def _kalman_update(self, cx: float, cy: float) -> None:
        """Обновляет Kalman измеренным центроидом."""
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self._kalman.correct(measurement)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _detect_polarity(self, frame: ProcessedFrame, bbox: BoundingBox) -> str:
        """Определяет, является ли цель горячей (ярче) или холодной (темнее) своего фона."""
        gray = frame.gray
        clamped = bbox.clamp(gray.shape)
        obj_patch = gray[clamped.y:clamped.y2, clamped.x:clamped.x2]
        if obj_patch.size == 0:
            return "hot"

        margin = max(4, min(clamped.width, clamped.height))
        outer = clamped.pad(margin, margin).clamp(gray.shape)
        outer_patch = gray[outer.y:outer.y2, outer.x:outer.x2]

        # Маска кольца (исключаем внутренний bbox из фона)
        ring_mask = np.ones(outer_patch.shape, dtype=bool)
        iy1 = clamped.y - outer.y
        ix1 = clamped.x - outer.x
        iy2 = iy1 + clamped.height
        ix2 = ix1 + clamped.width
        ring_mask[iy1:iy2, ix1:ix2] = False
        bg_values = outer_patch[ring_mask]
        if bg_values.size == 0:
            return "hot"

        bg_mean = float(np.mean(bg_values))
        obj_mean = float(np.mean(obj_patch))
        return "hot" if obj_mean >= bg_mean else "cold"

    def _measure_sharpness(self, frame: ProcessedFrame) -> float:
        """Оценивает резкость центральной части кадра через Лапласиан."""
        gray = frame.gray
        h, w = gray.shape[:2]
        roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.08):int(w * 0.92)]
        if roi.size == 0:
            roi = gray
        lap = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
        return float(np.percentile(np.abs(lap), 90))

    def _update_blur_state(self, frame: ProcessedFrame) -> bool:
        """Определяет, замутнён ли текущий кадр. Возвращает True при деградации."""
        if not self.config.blur_hold_enabled:
            self._blur_hold_frames = 0
            return False

        sharpness = self._measure_sharpness(frame)
        if self._sharpness_baseline is None:
            self._sharpness_baseline = max(sharpness, 1e-6)
            return False

        baseline = max(self._sharpness_baseline, 1e-6)
        degraded = sharpness <= baseline * self.config.blur_sharpness_drop_ratio
        if degraded:
            self._blur_hold_frames += 1
        else:
            self._blur_hold_frames = max(0, self._blur_hold_frames - 1)
        return degraded

    def _update_sharpness_baseline(self, frame: ProcessedFrame) -> None:
        """Медленно адаптирует базовую резкость по хорошим кадрам."""
        sharpness = self._measure_sharpness(frame)
        if self._sharpness_baseline is None:
            self._sharpness_baseline = max(sharpness, 1e-6)
            return
        alpha = 0.05
        self._sharpness_baseline = self._sharpness_baseline * (1 - alpha) + sharpness * alpha

    def _go_idle(self) -> TrackSnapshot:
        """Переводит трекер в IDLE после исчерпания лимита потери."""
        self._state = TrackerState.IDLE
        self._bbox = None
        self._message = f"Target #{self._track_id} lost, click again"
        self._track_id = None
        self._kalman_initialized = False
        return self.snapshot(GlobalMotion())
