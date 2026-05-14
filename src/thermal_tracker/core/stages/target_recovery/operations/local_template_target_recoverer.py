"""Локальный recoverer на базе шаблонов цели.

Стратегия повторяет принцип, который сейчас встроен в гибридный single-target
tracker: ведутся два шаблона цели (долгий и адаптивный), при потере цель ищется
локальной template-корреляцией в расширенной зоне поиска вокруг последнего
известного bbox с перебором нескольких масштабов. Возвращается bbox с лучшим
score, если он не ниже ``match_threshold``.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from thermal_tracker.core.domain.models import BoundingBox, ProcessedFrame
from thermal_tracker.core.stages.frame_stabilization.result import FrameStabilizerResult
from .base_target_recoverer import BaseReacquirer


def _safe_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Меняет размер изображения и не даёт получить нулевую ширину или высоту."""

    width, height = size
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _crop_gray(frame_gray: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    """Вырезает gray-патч и возвращает ``None``, если он выродился."""

    clamped = bbox.clamp(frame_gray.shape)
    if clamped.width <= 1 or clamped.height <= 1:
        return None
    return frame_gray[clamped.y:clamped.y2, clamped.x:clamped.x2]


class LocalTemplateReacquirer(BaseReacquirer):
    """Локальный recoverer по шаблону цели в расширенной зоне поиска."""

    def __init__(
        self,
        search_padding: int = 60,
        search_padding_growth: int = 8,
        scales: Sequence[float] = (0.84, 0.94, 1.0, 1.08, 1.18),
        match_threshold: float = 0.5,
        template_alpha: float = 0.12,
    ) -> None:
        self._search_padding = max(0, search_padding)
        self._search_padding_growth = max(0, search_padding_growth)
        self._scales = tuple(float(scale) for scale in scales) or (1.0,)
        self._match_threshold = float(match_threshold)
        self._template_alpha = float(np.clip(template_alpha, 0.0, 1.0))

        self._canonical_size: tuple[int, int] | None = None
        self._long_term_template: np.ndarray | None = None
        self._adaptive_template: np.ndarray | None = None

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Инициализирует или адаптивно обновляет шаблоны цели."""

        gray_patch = _crop_gray(frame.gray, bbox)
        if gray_patch is None:
            return

        if self._canonical_size is None or self._long_term_template is None or self._adaptive_template is None:
            self._canonical_size = (gray_patch.shape[1], gray_patch.shape[0])
            self._long_term_template = gray_patch.copy()
            self._adaptive_template = gray_patch.copy()
            return

        resized = _safe_resize(gray_patch, self._canonical_size)
        alpha = self._template_alpha
        adaptive_float = self._adaptive_template.astype(np.float32)
        new_float = resized.astype(np.float32)
        self._adaptive_template = cv2.addWeighted(
            new_float,
            alpha,
            adaptive_float,
            1.0 - alpha,
            0.0,
        ).astype(np.uint8)

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: FrameStabilizerResult,
        lost_frames: int = 0,
    ) -> BoundingBox | None:
        """Ищет цель в расширенной зоне поиска вокруг ``last_bbox``.

        Зона поиска растёт линейно: ``search_padding + lost_frames * search_padding_growth``.
        Это даёт recoverer-у шанс догнать цель, ушедшую дальше из узкой стартовой зоны.
        """

        if self._canonical_size is None or self._adaptive_template is None or self._long_term_template is None:
            return None

        frame_gray = frame.gray
        shifted_bbox = self._shift_bbox_by_motion(last_bbox, motion).clamp(frame_gray.shape)
        effective_padding = self._search_padding + max(0, int(lost_frames)) * self._search_padding_growth
        search_region = shifted_bbox.pad(effective_padding, effective_padding).clamp(frame_gray.shape)
        search_patch = _crop_gray(frame_gray, search_region)
        if search_patch is None:
            return None

        canonical_w, canonical_h = self._canonical_size
        best_score = -1.0
        best_bbox: BoundingBox | None = None
        for scale in self._scales:
            template_w = max(2, int(round(canonical_w * scale)))
            template_h = max(2, int(round(canonical_h * scale)))
            if template_w >= search_patch.shape[1] or template_h >= search_patch.shape[0]:
                continue

            adaptive_template = _safe_resize(self._adaptive_template, (template_w, template_h))
            long_term_template = _safe_resize(self._long_term_template, (template_w, template_h))
            score, location = self._best_match(search_patch, adaptive_template, long_term_template)
            if score <= best_score:
                continue
            best_score = score
            best_bbox = BoundingBox(
                x=search_region.x + int(location[0]),
                y=search_region.y + int(location[1]),
                width=template_w,
                height=template_h,
            ).clamp(frame_gray.shape)

        if best_bbox is None or best_score < self._match_threshold:
            return None
        return best_bbox

    def reset(self) -> None:
        """Полностью забывает шаблоны и канонический размер."""

        self._canonical_size = None
        self._long_term_template = None
        self._adaptive_template = None

    @staticmethod
    def _shift_bbox_by_motion(bbox: BoundingBox, motion: FrameStabilizerResult) -> BoundingBox:
        """Корректирует bbox на сдвиг камеры, если оценка движения валидна."""

        if not motion.valid:
            return bbox
        cx, cy = bbox.center
        return BoundingBox.from_center(
            cx + float(motion.dx),
            cy + float(motion.dy),
            bbox.width,
            bbox.height,
        )

    @staticmethod
    def _best_match(
        search_patch: np.ndarray,
        adaptive_template: np.ndarray,
        long_term_template: np.ndarray,
    ) -> tuple[float, tuple[int, int]]:
        """Считает максимальный template-score из двух шаблонов."""

        adaptive_response = cv2.matchTemplate(search_patch, adaptive_template, cv2.TM_CCOEFF_NORMED)
        long_term_response = cv2.matchTemplate(search_patch, long_term_template, cv2.TM_CCOEFF_NORMED)

        _, adaptive_score, _, adaptive_location = cv2.minMaxLoc(adaptive_response)
        _, long_term_score, _, long_term_location = cv2.minMaxLoc(long_term_response)

        if float(adaptive_score) >= float(long_term_score):
            return float(adaptive_score), adaptive_location
        return float(long_term_score), long_term_location
