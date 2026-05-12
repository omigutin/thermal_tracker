"""IRST-recoverer на базе локального контраста.

Вместо сопоставления шаблонов ищет контрастный blob в расширенной зоне вокруг
последней известной позиции цели. Запоминает полярность (hot/cold) и
канонический размер blob — это единственные «шаблонные» данные recoverer-а.

Используется в связке с IrstSingleTargetTracker через ManualClickTrackingPipeline:
pipeline вызывает remember() на каждом уверенном TRACKING-кадре и reacquire()
когда трекер перешёл в SEARCHING и потеря превысила min_lost_frames.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame
from .base_target_recoverer import BaseReacquirer


class IrstContrastRecoverer(BaseReacquirer):
    """Повторный захват цели через детектор локального контраста.

    Параметры задаются напрямую конструктором — manager передаёт их из
    TargetRecoveryConfig (search_padding, search_padding_growth и т.д.).
    """

    def __init__(
        self,
        search_padding: int = 30,
        search_padding_growth: int = 4,
        contrast_threshold: float = 10.0,
        min_blob_area: int = 1,
        max_blob_area: int = 300,
        match_threshold: float = 0.0,  # не используется, оставлен для совместимости контракта
    ) -> None:
        self._search_padding = max(0, search_padding)
        self._search_padding_growth = max(0, search_padding_growth)
        self._contrast_threshold = contrast_threshold
        self._min_blob_area = min_blob_area
        self._max_blob_area = max_blob_area

        # Память о последней надёжно сопровождавшейся цели
        self._polarity: str = "hot"          # "hot" или "cold"
        self._canonical_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Контракт BaseReacquirer
    # ------------------------------------------------------------------

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Запоминает полярность и размер цели по уверенному TRACKING-кадру."""
        self._canonical_size = (bbox.width, bbox.height)
        self._polarity = self._detect_polarity(frame, bbox)

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
        lost_frames: int = 0,
    ) -> BoundingBox | None:
        """Ищет цель в расширенной зоне вокруг last_bbox по локальному контрасту.

        Зона поиска растёт линейно: search_padding + lost_frames * search_padding_growth.
        Возвращает bbox лучшего кандидата или None, если ничего не найдено.
        """
        # Корректируем last_bbox на сдвиг камеры (если оценка движения валидна)
        shifted_bbox = self._shift_by_motion(last_bbox, motion)

        # Расширяем зону поиска
        effective_padding = self._search_padding + max(0, lost_frames) * self._search_padding_growth
        search_region = shifted_bbox.pad(effective_padding, effective_padding).clamp(frame.gray.shape)

        candidates = self._find_candidates(frame, search_region)
        if not candidates:
            return None

        # Возвращаем кандидата, ближайшего к центру last_bbox
        ref_cx, ref_cy = shifted_bbox.center
        best_bbox, _ = min(
            candidates,
            key=lambda c: math.hypot(c[0].center[0] - ref_cx, c[0].center[1] - ref_cy),
        )
        return best_bbox

    def reset(self) -> None:
        """Сбрасывает запомненные данные о цели."""
        self._canonical_size = None
        self._polarity = "hot"

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        frame: ProcessedFrame,
        search_region: BoundingBox,
    ) -> list[tuple[BoundingBox, float]]:
        """Находит контрастные blob-ы в зоне поиска с учётом полярности цели."""
        gray = frame.gray
        contrast_map = self._compute_contrast_map(gray)

        # Маска зоны поиска
        mask = np.zeros(contrast_map.shape, dtype=np.float32)
        r = search_region.clamp(gray.shape)
        mask[r.y:r.y2, r.x:r.x2] = 1.0
        masked = contrast_map * mask

        # Пороговая фильтрация
        binary = (masked >= self._contrast_threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        candidates: list[tuple[BoundingBox, float]] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (self._min_blob_area <= area <= self._max_blob_area):
                continue

            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            # Размер bbox: используем канонический размер если известен
            if self._canonical_size is not None:
                cw, ch = self._canonical_size
            else:
                cw = max(int(stats[i, cv2.CC_STAT_WIDTH]), 6)
                ch = max(int(stats[i, cv2.CC_STAT_HEIGHT]), 6)

            bbox = BoundingBox.from_center(cx, cy, cw, ch).clamp(gray.shape)
            blob_pixels = masked[labels == i]
            score = float(np.mean(blob_pixels)) / 255.0 if blob_pixels.size > 0 else 0.0
            candidates.append((bbox, score))

        candidates.sort(key=lambda c: -c[1])
        return candidates

    def _compute_contrast_map(self, gray: np.ndarray) -> np.ndarray:
        """Вычисляет карту локального контраста (hot + cold)."""
        gray_f = gray.astype(np.float32)

        inner_kernel = np.ones((3, 3), np.uint8)
        local_max = cv2.dilate(gray, inner_kernel).astype(np.float32)
        local_min = cv2.erode(gray, inner_kernel).astype(np.float32)
        background = cv2.blur(gray_f, (11, 11))

        hot = local_max - background
        cold = background - local_min
        contrast = np.maximum(hot, cold)
        return np.clip(contrast, 0.0, 255.0)

    def _detect_polarity(self, frame: ProcessedFrame, bbox: BoundingBox) -> str:
        """Определяет полярность цели (hot/cold) по последнему известному bbox."""
        gray = frame.gray
        clamped = bbox.clamp(gray.shape)
        obj = gray[clamped.y:clamped.y2, clamped.x:clamped.x2]
        if obj.size == 0:
            return "hot"
        margin = max(4, min(clamped.width, clamped.height))
        outer = clamped.pad(margin, margin).clamp(gray.shape)
        outer_patch = gray[outer.y:outer.y2, outer.x:outer.x2]
        ring_mask = np.ones(outer_patch.shape, dtype=bool)
        iy1 = clamped.y - outer.y
        ix1 = clamped.x - outer.x
        ring_mask[iy1:iy1 + clamped.height, ix1:ix1 + clamped.width] = False
        bg = outer_patch[ring_mask]
        if bg.size == 0:
            return "hot"
        return "hot" if float(np.mean(obj)) >= float(np.mean(bg)) else "cold"

    @staticmethod
    def _shift_by_motion(bbox: BoundingBox, motion: GlobalMotion) -> BoundingBox:
        """Смещает bbox на вектор глобального движения камеры."""
        if not motion.valid:
            return bbox
        cx, cy = bbox.center
        return BoundingBox.from_center(
            cx + float(motion.dx),
            cy + float(motion.dy),
            bbox.width,
            bbox.height,
        )
