"""Выделение объекта по одному клику.

Идея простая:
- пользователь кликнул по цели;
- сначала находим уверенное горячее или холодное ядро;
- потом пытаемся расширить его до целого объекта.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ...config import ClickSelectionConfig
from ...domain.models import BoundingBox, ProcessedFrame, SelectionResult
from .base_target_selector import BaseClickInitializer


@dataclass
class _LocalPatch:
    """Локальный кусок кадра вокруг клика или текущего бокса."""

    image: np.ndarray
    origin_x: int
    origin_y: int
    local_x: int
    local_y: int


class ClickTargetSelector(BaseClickInitializer):
    implementation_name = "hybrid_click"
    is_ready = True
    """Ищет объект вокруг точки и возвращает прямоугольник цели."""

    def __init__(self, config: ClickSelectionConfig) -> None:
        self.config = config

    def select(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> SelectionResult:
        """Находит цель вокруг точки.

        Если `expected_bbox` пустой, считаем, что это новый клик.
        Если он задан, то это аккуратное уточнение уже найденной цели.
        """

        attempt_radii = self._build_attempt_radii(expected_bbox)
        fallback: SelectionResult | None = None

        for attempt_index, radius in enumerate(attempt_radii):
            patch = self._extract_patch(frame.normalized, point, expected_bbox, radius)
            if expected_bbox is None:
                patch = self._snap_patch_point(patch)
            mask, polarity = self._build_mask(patch.image, patch.local_x, patch.local_y)
            component_bbox, confidence = self._extract_component(mask, patch, expected_bbox)
            if component_bbox is None and expected_bbox is None:
                component_bbox, confidence = self._extract_component(
                    self._build_score_mask(patch.image, patch.local_x, patch.local_y),
                    patch,
                    expected_bbox,
                )

            if component_bbox is None:
                continue

            selection = SelectionResult(
                bbox=component_bbox.clamp(frame.bgr.shape),
                confidence=confidence,
                polarity=polarity,
            )
            if expected_bbox is None:
                selection = self._expand_selection(patch, selection)
            fallback = selection

            if expected_bbox is None and attempt_index + 1 < len(attempt_radii):
                if self._touches_patch_border(selection.bbox, patch):
                    continue

            return selection

        if fallback is not None:
            return fallback

        return self._fallback(point, frame.bgr.shape, expected_bbox)

    def refine(self, frame: ProcessedFrame, bbox: BoundingBox) -> SelectionResult | None:
        """Слегка уточняет уже найденный бокс, но без агрессивных прыжков."""

        selection = self.select(
            frame=frame,
            point=(int(bbox.center[0]), int(bbox.center[1])),
            expected_bbox=bbox,
        )

        if selection.bbox.area <= 0:
            return None
        if selection.bbox.intersection_over_union(bbox) < 0.15:
            return None
        if selection.bbox.width < max(4, self.config.min_component_area // 2):
            return None
        if selection.bbox.width > int(bbox.width * self.config.max_refine_growth):
            return None
        if selection.bbox.height > int(bbox.height * self.config.max_refine_growth):
            return None
        if selection.bbox.area > int(bbox.area * (self.config.max_refine_growth ** 2)):
            return None
        return selection

    def _snap_patch_point(self, patch: _LocalPatch) -> _LocalPatch:
        """Чуть сдвигает клик к ближайшему выразительному пикселю цели."""

        radius = max(6, self.config.local_window_radius * 2 + 2)
        x1 = max(0, patch.local_x - radius)
        y1 = max(0, patch.local_y - radius)
        x2 = min(patch.image.shape[1], patch.local_x + radius + 1)
        y2 = min(patch.image.shape[0], patch.local_y + radius + 1)
        window = patch.image[y1:y2, x1:x2]
        if window.size == 0:
            return patch

        blurred = cv2.GaussianBlur(window, (5, 5), 0)
        local_median = float(np.median(blurred))
        yy, xx = np.indices(blurred.shape)
        distance = np.sqrt((xx - (patch.local_x - x1)) ** 2 + (yy - (patch.local_y - y1)) ** 2)
        deviation = np.abs(blurred.astype(np.float32) - local_median)
        score = deviation - distance * 1.6

        best_y, best_x = np.unravel_index(int(np.argmax(score)), score.shape)
        if float(deviation[best_y, best_x]) < self.config.min_object_contrast:
            return patch

        return _LocalPatch(
            image=patch.image,
            origin_x=patch.origin_x,
            origin_y=patch.origin_y,
            local_x=int(x1 + best_x),
            local_y=int(y1 + best_y),
        )

    def _expand_selection(self, patch: _LocalPatch, selection: SelectionResult) -> SelectionResult:
        """Пытается превратить яркое ядро в бокс всего объекта."""

        local_seed = BoundingBox(
            x=selection.bbox.x - patch.origin_x,
            y=selection.bbox.y - patch.origin_y,
            width=selection.bbox.width,
            height=selection.bbox.height,
        ).clamp(patch.image.shape)
        if local_seed.area <= 0:
            return selection

        margin = max(
            self.config.expansion_margin,
            int(round(max(local_seed.width, local_seed.height) * 0.5)),
        )
        work_bbox = local_seed.pad(margin, margin).clamp(patch.image.shape)
        work_patch = patch.image[work_bbox.y:work_bbox.y2, work_bbox.x:work_bbox.x2]
        if work_patch.size == 0:
            return selection

        seed_x1 = local_seed.x - work_bbox.x
        seed_y1 = local_seed.y - work_bbox.y
        seed_x2 = seed_x1 + local_seed.width
        seed_y2 = seed_y1 + local_seed.height
        seed_mask = np.zeros(work_patch.shape, dtype=np.uint8)
        seed_mask[seed_y1:seed_y2, seed_x1:seed_x2] = 255

        object_values = work_patch[seed_mask > 0]
        if object_values.size == 0:
            return selection

        background_mask = np.ones(work_patch.shape, dtype=bool)
        background_inner = BoundingBox(
            x=seed_x1,
            y=seed_y1,
            width=local_seed.width,
            height=local_seed.height,
        ).pad(self.config.background_ring, self.config.background_ring).clamp(work_patch.shape)
        background_mask[background_inner.y:background_inner.y2, background_inner.x:background_inner.x2] = False
        background_values = work_patch[background_mask]
        if background_values.size < 10:
            return selection

        object_level = float(np.median(object_values))
        background_level = float(np.median(background_values))
        contrast = object_level - background_level
        if abs(contrast) < self.config.min_object_contrast:
            return selection

        threshold = background_level + contrast * self.config.foreground_fraction
        if selection.polarity == "cold":
            object_mask = (work_patch.astype(np.float32) <= threshold).astype(np.uint8) * 255
        else:
            object_mask = (work_patch.astype(np.float32) >= threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_support = cv2.dilate(seed_mask, kernel, iterations=1)

        _, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask)
        overlapping_labels = np.unique(labels[seed_support > 0])
        overlapping_labels = overlapping_labels[overlapping_labels > 0]
        if overlapping_labels.size == 0:
            return selection

        best_label = max(overlapping_labels.tolist(), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
        area = int(stats[best_label, cv2.CC_STAT_AREA])
        if area <= selection.bbox.area:
            return selection
        if area > int(work_patch.shape[0] * work_patch.shape[1] * self.config.max_expanded_fill):
            return selection
        if area > int(selection.bbox.area * self.config.max_expansion_ratio):
            return selection

        x = int(stats[best_label, cv2.CC_STAT_LEFT])
        y = int(stats[best_label, cv2.CC_STAT_TOP])
        w = int(stats[best_label, cv2.CC_STAT_WIDTH])
        h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

        expanded_bbox = BoundingBox(
            x=patch.origin_x + work_bbox.x + x - self.config.padding,
            y=patch.origin_y + work_bbox.y + y - self.config.padding,
            width=w + self.config.padding * 2,
            height=h + self.config.padding * 2,
        )
        return SelectionResult(
            bbox=expanded_bbox,
            confidence=max(selection.confidence, min(1.0, area / max(self.config.min_component_area, 1))),
            polarity=selection.polarity,
        )

    def _extract_patch(
        self,
        image: np.ndarray,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None,
        radius_override: int | None = None,
    ) -> _LocalPatch:
        """Вырезает локальный патч вокруг точки интереса."""

        frame_h, frame_w = image.shape[:2]
        radius = radius_override if radius_override is not None else self.config.search_radius
        if expected_bbox is not None and radius_override is None:
            radius = max(
                radius // 2,
                int(max(expected_bbox.width, expected_bbox.height) * 0.8) + self.config.padding,
            )

        x = int(np.clip(point[0], 0, frame_w - 1))
        y = int(np.clip(point[1], 0, frame_h - 1))
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(frame_w, x + radius + 1)
        y2 = min(frame_h, y + radius + 1)
        return _LocalPatch(
            image=image[y1:y2, x1:x2],
            origin_x=x1,
            origin_y=y1,
            local_x=x - x1,
            local_y=y - y1,
        )

    def _build_mask(self, patch: np.ndarray, click_x: int, click_y: int) -> tuple[np.ndarray, str]:
        """Строит стартовую маску вокруг кликнутого пикселя."""

        clicked_value = int(patch[click_y, click_x])
        median_value = float(np.median(patch))
        tolerance = self._estimate_tolerance(patch, click_x, click_y)

        patch_int = patch.astype(np.int16)
        similarity_mask = np.abs(patch_int - clicked_value) <= tolerance
        hot_object = clicked_value >= median_value

        if hot_object:
            intensity_mask = patch >= (clicked_value - tolerance)
            polarity = "hot"
        else:
            intensity_mask = patch <= (clicked_value + tolerance)
            polarity = "cold"

        mask = np.logical_and(similarity_mask, intensity_mask).astype(np.uint8) * 255
        if mask[click_y, click_x] == 0:
            mask = similarity_mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask, polarity

    def _build_score_mask(self, patch: np.ndarray, click_x: int, click_y: int) -> np.ndarray:
        """Резервный вариант маски, если обычная похожесть не сработала."""

        clicked_value = float(patch[click_y, click_x])
        tolerance = float(self._estimate_tolerance(patch, click_x, click_y))

        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0.0, 1.0, cv2.NORM_MINMAX)

        yy, xx = np.indices(patch.shape)
        distance = np.sqrt((xx - click_x) ** 2 + (yy - click_y) ** 2)
        distance /= max(float(max(patch.shape)), 1.0)

        difference = np.abs(patch.astype(np.float32) - clicked_value) / max(tolerance, 1.0)
        score = difference * 0.6 + gradient * 0.3 + distance * 0.1
        mask = (score <= 0.95).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _extract_component(
        self,
        mask: np.ndarray,
        patch: _LocalPatch,
        expected_bbox: BoundingBox | None,
    ) -> tuple[BoundingBox | None, float]:
        """Берёт компоненту связности, к которой относится клик."""

        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clicked_label = int(labels[patch.local_y, patch.local_x])

        if clicked_label == 0:
            clicked_label = self._nearest_label(labels, patch.local_x, patch.local_y)
            if clicked_label == 0:
                return None, 0.0

        area = int(stats[clicked_label, cv2.CC_STAT_AREA])
        patch_area = mask.shape[0] * mask.shape[1]
        if area < self.config.min_component_area:
            return None, 0.0
        if area > patch_area * self.config.max_component_fill:
            return None, 0.0

        x = int(stats[clicked_label, cv2.CC_STAT_LEFT])
        y = int(stats[clicked_label, cv2.CC_STAT_TOP])
        w = int(stats[clicked_label, cv2.CC_STAT_WIDTH])
        h = int(stats[clicked_label, cv2.CC_STAT_HEIGHT])

        if expected_bbox is None:
            split_bbox = self._split_large_component(mask, labels, clicked_label, patch.local_x, patch.local_y)
            if split_bbox is not None:
                x, y, w, h = split_bbox.to_xywh()

        max_patch_width = int(mask.shape[1] * self.config.max_patch_span_ratio)
        max_patch_height = int(mask.shape[0] * self.config.max_patch_span_ratio)
        if expected_bbox is None:
            if w > max_patch_width or h > max_patch_height:
                return None, 0.0
        else:
            if w > int(expected_bbox.width * self.config.max_refine_growth):
                return None, 0.0
            if h > int(expected_bbox.height * self.config.max_refine_growth):
                return None, 0.0

        bbox = BoundingBox(
            x=patch.origin_x + x - self.config.padding,
            y=patch.origin_y + y - self.config.padding,
            width=w + self.config.padding * 2,
            height=h + self.config.padding * 2,
        )

        confidence = max(0.1, min(1.0, area / max(self.config.min_component_area, 1)))
        return bbox, confidence

    def _split_large_component(
        self,
        mask: np.ndarray,
        labels: np.ndarray,
        clicked_label: int,
        click_x: int,
        click_y: int,
    ) -> BoundingBox | None:
        """Пытается расколоть слишком крупную компоненту на отдельные объекты."""

        component_mask = (labels == clicked_label).astype(np.uint8) * 255
        original_area = int(np.count_nonzero(component_mask))
        if original_area < self.config.min_component_area * 4:
            return None

        best_bbox: BoundingBox | None = None
        best_area = original_area
        kernels = ((3, 3), (5, 5))

        for kernel_size in kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            eroded = cv2.erode(component_mask, kernel, iterations=1)
            if int(np.count_nonzero(eroded)) < self.config.min_component_area:
                continue

            _, split_labels, split_stats, _ = cv2.connectedComponentsWithStats(eroded)
            candidate_labels = [
                label
                for label in range(1, split_stats.shape[0])
                if int(split_stats[label, cv2.CC_STAT_AREA]) >= self.config.min_component_area
            ]
            if len(candidate_labels) < 2:
                continue

            best_candidate: tuple[float, BoundingBox, int] | None = None
            for label in candidate_labels:
                sub_mask = (split_labels == label).astype(np.uint8) * 255
                restored = cv2.dilate(sub_mask, kernel, iterations=1)
                points = np.column_stack(np.where(restored > 0))
                if points.size == 0:
                    continue

                left = int(np.min(points[:, 1]))
                top = int(np.min(points[:, 0]))
                right = int(np.max(points[:, 1])) + 1
                bottom = int(np.max(points[:, 0])) + 1
                bbox = BoundingBox(left, top, right - left, bottom - top)
                area = int(np.count_nonzero(restored))

                if restored[click_y, click_x] > 0:
                    distance_score = -1.0
                else:
                    center_x, center_y = bbox.center
                    distance_score = float((center_x - click_x) ** 2 + (center_y - click_y) ** 2)

                candidate = (distance_score, bbox, area)
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

            if best_candidate is None:
                continue

            _, bbox, area = best_candidate
            if area < best_area * 0.88:
                best_bbox = bbox
                best_area = area

        return best_bbox

    def _build_attempt_radii(self, expected_bbox: BoundingBox | None) -> list[int]:
        """Готовит радиусы патча для одной или двух попыток выбора."""

        if expected_bbox is not None:
            radius = max(
                self.config.search_radius // 2,
                int(max(expected_bbox.width, expected_bbox.height) * 0.8) + self.config.padding,
            )
            return [radius]

        base_radius = self.config.search_radius
        retry_radius = min(
            self.config.max_retry_radius,
            int(round(base_radius * self.config.retry_scale)),
        )
        if retry_radius <= base_radius:
            return [base_radius]
        return [base_radius, retry_radius]

    def _local_window(self, patch: np.ndarray, click_x: int, click_y: int) -> np.ndarray:
        """Возвращает маленькое окно вокруг клика для локальной статистики."""

        radius = self.config.local_window_radius
        return patch[
            max(0, click_y - radius): click_y + radius + 1,
            max(0, click_x - radius): click_x + radius + 1,
        ]

    def _estimate_tolerance(self, patch: np.ndarray, click_x: int, click_y: int) -> int:
        """Оценивает, насколько далеко по яркости можно отходить от клика."""

        local = self._local_window(patch, click_x, click_y)
        local_std = float(np.std(local) + 1.0)
        tolerance = int(round(local_std * self.config.similarity_sigma + 4.0))
        tolerance = max(self.config.min_tolerance, tolerance)
        tolerance = min(self.config.max_tolerance, tolerance)
        return tolerance

    @staticmethod
    def _touches_patch_border(bbox: BoundingBox, patch: _LocalPatch, tolerance: int = 2) -> bool:
        """Проверяет, не упёрлась ли найденная область в край локального патча."""

        local_x1 = bbox.x - patch.origin_x
        local_y1 = bbox.y - patch.origin_y
        local_x2 = local_x1 + bbox.width
        local_y2 = local_y1 + bbox.height
        patch_h, patch_w = patch.image.shape[:2]
        return (
            local_x1 <= tolerance
            or local_y1 <= tolerance
            or local_x2 >= patch_w - tolerance
            or local_y2 >= patch_h - tolerance
        )

    @staticmethod
    def _nearest_label(labels: np.ndarray, x: int, y: int) -> int:
        """Находит ближайшую непустую компоненту, если клик попал на дырку."""

        window = labels[max(0, y - 2): y + 3, max(0, x - 2): x + 3]
        non_zero = window[window > 0]
        if non_zero.size == 0:
            return 0
        values, counts = np.unique(non_zero, return_counts=True)
        return int(values[np.argmax(counts)])

    def _fallback(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        expected_bbox: BoundingBox | None,
    ) -> SelectionResult:
        """Последний запасной вариант, когда нормальная сегментация не взлетела."""

        size = self.config.fallback_size
        if expected_bbox is not None:
            size = int(max(expected_bbox.width, expected_bbox.height))

        bbox = BoundingBox.from_center(point[0], point[1], size, size).clamp(frame_shape)
        return SelectionResult(bbox=bbox, confidence=0.05, polarity="fallback")
