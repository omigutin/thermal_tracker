from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .....domain.models import BoundingBox
from .contrast_component_patch import ContrastComponentPatch


class ContrastComponentPatchExtractorSettings(Protocol):
    """Описывает настройки, нужные для извлечения локального участка по клику."""

    # Базовый радиус локального участка вокруг точки клика.
    search_radius: int
    # Отступ вокруг найденной или ожидаемой области цели.
    padding: int
    # Радиус локального окна вокруг клика для оценки выразительности пикселя.
    local_window_radius: int
    # Минимальный контраст пикселя относительно окружения, чтобы считать его частью цели.
    min_object_contrast: float

    # Минимальный радиус окна для смещения клика к более выразительному пикселю.
    snap_min_radius: int
    # Множитель радиуса локального окна для поиска более выразительного пикселя.
    snap_radius_multiplier: float
    # Дополнительный запас радиуса при поиске более выразительного пикселя.
    snap_radius_padding: int
    # Размер ядра размытия перед оценкой выразительности пикселей.
    snap_blur_kernel: int
    # Штраф за удалённость пикселя от исходной точки клика.
    snap_distance_weight: float

    # Доля размера ожидаемого bbox для расчёта радиуса уточняющего патча.
    refine_patch_bbox_scale: float
    # Минимальная доля базового радиуса при уточнении уже известной цели.
    refine_patch_min_radius_ratio: float


@dataclass(slots=True)
class ContrastComponentPatchExtractor:
    """Вырезает локальный участок кадра и уточняет точку клика внутри него."""

    config: ContrastComponentPatchExtractorSettings

    def extract_patch(
        self,
        image: np.ndarray,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None,
        radius_override: int | None = None,
    ) -> ContrastComponentPatch:
        """
            Вырезать локальный участок вокруг точки интереса.
            Если `expected_bbox` задан, участок считается уточняющим: радиус
            подбирается по размеру уже известной цели, чтобы не захватывать лишнее окружение.
        """
        frame_height, frame_width = image.shape[:2]
        radius = self._resolve_patch_radius(expected_bbox=expected_bbox, radius_override=radius_override)

        # Клик зажимается в границы кадра, чтобы дальнейшая нарезка patch
        # не зависела от качества внешней валидации координат.
        x = int(np.clip(point[0], 0, frame_width - 1))
        y = int(np.clip(point[1], 0, frame_height - 1))

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(frame_width, x + radius + 1)
        y2 = min(frame_height, y + radius + 1)

        return ContrastComponentPatch(image=image[y1:y2, x1:x2], origin_x=x1, origin_y=y1, local_x=x - x1, local_y=y - y1)

    def snap_patch_point(self, patch: ContrastComponentPatch) -> ContrastComponentPatch:
        """
            Сдвинуть точку клика к ближайшему выразительному пикселю цели.
            Это нужно, когда пользователь кликнул не в яркое/холодное ядро цели, а рядом с ним.
            Мы ищем локальный максимум отклонения от медианы,
            но штрафуем дальние пиксели, чтобы точка не прыгала слишком далеко.
        """
        radius = self._resolve_snap_radius()

        x1 = max(0, patch.local_x - radius)
        y1 = max(0, patch.local_y - radius)
        x2 = min(patch.width, patch.local_x + radius + 1)
        y2 = min(patch.height, patch.local_y + radius + 1)

        window = patch.image[y1:y2, x1:x2]
        if window.size == 0:
            return patch

        blurred = cv2.GaussianBlur(window, (self.config.snap_blur_kernel, self.config.snap_blur_kernel), 0)
        clicked_value = float(patch.image[patch.local_y, patch.local_x])
        local_median = float(np.median(blurred))

        # Если кликнутый пиксель уже достаточно контрастный, не двигаем точку:
        # лишнее смещение может ухудшить точный ручной выбор.
        if abs(clicked_value - local_median) >= self.config.min_object_contrast:
            return patch

        deviation = self._build_deviation_map(blurred=blurred, clicked_value=clicked_value, local_median=local_median)
        distance = self._build_distance_map(shape=blurred.shape, center_x=patch.local_x - x1, center_y=patch.local_y - y1)

        score = deviation - distance * self.config.snap_distance_weight
        best_y, best_x = np.unravel_index(int(np.argmax(score)), score.shape)

        # Если даже лучший соседний пиксель недостаточно выразительный,
        # оставляем исходную точку, чтобы не привнести случайный шум.
        if float(deviation[best_y, best_x]) < self.config.min_object_contrast:
            return patch

        return ContrastComponentPatch(
            image=patch.image,
            origin_x=patch.origin_x,
            origin_y=patch.origin_y,
            local_x=int(x1 + best_x),
            local_y=int(y1 + best_y),
        )

    def _resolve_patch_radius(self, expected_bbox: BoundingBox | None, radius_override: int | None) -> int:
        """Рассчитать радиус локального участка."""
        if radius_override is not None:
            return radius_override

        radius = self.config.search_radius

        if expected_bbox is None:
            return radius

        min_refine_radius = int(round(radius * self.config.refine_patch_min_radius_ratio))
        bbox_based_radius = (
            int(max(expected_bbox.width, expected_bbox.height) * self.config.refine_patch_bbox_scale)
            + self.config.padding
        )

        return max(min_refine_radius, bbox_based_radius)

    def _resolve_snap_radius(self) -> int:
        """Рассчитать радиус поиска более выразительного пикселя."""
        radius = int(round(self.config.local_window_radius * self.config.snap_radius_multiplier))
        radius += self.config.snap_radius_padding
        return max(self.config.snap_min_radius, radius)

    @staticmethod
    def _build_distance_map(shape: tuple[int, int], center_x: int, center_y: int) -> np.ndarray:
        """Построить карту расстояний от исходной точки клика."""
        yy, xx = np.indices(shape)
        return np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    def _build_deviation_map(self, blurred: np.ndarray, clicked_value: float, local_median: float) -> np.ndarray:
        """Построить карту полезного отклонения яркости от локальной медианы."""
        raw_deviation = blurred.astype(np.float32) - local_median
        clicked_deviation = clicked_value - local_median

        # Если направление контраста клика понятно, ищем пиксели той же полярности.
        # Если клик слабый, используем абсолютное отклонение и не угадываем сторону.
        if abs(clicked_deviation) < self.config.min_object_contrast:
            return np.abs(raw_deviation)

        if clicked_deviation >= 0:
            return np.maximum(raw_deviation, 0.0)

        return np.maximum(-raw_deviation, 0.0)
