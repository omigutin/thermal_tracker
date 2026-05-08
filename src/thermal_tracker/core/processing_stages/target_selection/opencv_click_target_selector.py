"""Выделение объекта по одному клику.

Идея простая:
- пользователь кликнул по цели;
- сначала находим уверенное горячее или холодное ядро;
- потом пытаемся расширить его до целого объекта.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...config import ClickSelectionConfig
from ...domain.models import BoundingBox, ProcessedFrame, SelectionResult
from .base_target_selector import BaseClickInitializer
from .local_patch import LocalPatch


class ClickTargetSelector(BaseClickInitializer):
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
                contrast_selection = self._select_contrast_component(patch, frame.bgr.shape)
                if contrast_selection is not None:
                    return contrast_selection
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

    def _snap_patch_point(self, patch: LocalPatch) -> LocalPatch:
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
        clicked_value = float(patch.image[patch.local_y, patch.local_x])
        local_median = float(np.median(blurred))
        if abs(clicked_value - local_median) >= self.config.min_object_contrast:
            return patch

        yy, xx = np.indices(blurred.shape)
        distance = np.sqrt((xx - (patch.local_x - x1)) ** 2 + (yy - (patch.local_y - y1)) ** 2)
        raw_deviation = blurred.astype(np.float32) - local_median
        clicked_deviation = clicked_value - local_median
        if abs(clicked_deviation) >= self.config.min_object_contrast:
            if clicked_deviation >= 0:
                deviation = np.maximum(raw_deviation, 0.0)
            else:
                deviation = np.maximum(-raw_deviation, 0.0)
        else:
            deviation = np.abs(raw_deviation)
        score = deviation - distance * 1.6

        best_y, best_x = np.unravel_index(int(np.argmax(score)), score.shape)
        if float(deviation[best_y, best_x]) < self.config.min_object_contrast:
            return patch

        return LocalPatch(
            image=patch.image,
            origin_x=patch.origin_x,
            origin_y=patch.origin_y,
            local_x=int(x1 + best_x),
            local_y=int(y1 + best_y),
        )

    def _expand_selection(self, patch: LocalPatch, selection: SelectionResult) -> SelectionResult:
        """Пытается превратить яркое ядро в бокс всего объекта."""

        local_seed = BoundingBox(
            x=selection.bbox.x - patch.origin_x,
            y=selection.bbox.y - patch.origin_y,
            width=selection.bbox.width,
            height=selection.bbox.height,
        ).clamp(patch.image.shape)
        if local_seed.area <= 0:
            return selection
        if local_seed.area > 3000:
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
        allowed_expansion_ratio = self._allowed_expansion_ratio(selection.bbox.area)
        if area > int(selection.bbox.area * allowed_expansion_ratio):
            return selection

        x = int(stats[best_label, cv2.CC_STAT_LEFT])
        y = int(stats[best_label, cv2.CC_STAT_TOP])
        w = int(stats[best_label, cv2.CC_STAT_WIDTH])
        h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
        expanded_width = w + self.config.padding * 2
        expanded_height = h + self.config.padding * 2
        allowed_side_ratio = self._allowed_expansion_side_ratio(selection.bbox.area)
        if expanded_width > int(selection.bbox.width * allowed_side_ratio):
            return selection
        if expanded_height > int(selection.bbox.height * allowed_side_ratio):
            return selection
        expanded_aspect = max(
            expanded_width / max(float(expanded_height), 1.0),
            expanded_height / max(float(expanded_width), 1.0),
        )
        small_target_mode = self.config.fallback_size <= 18 and self.config.max_expansion_ratio <= 2.5
        expected_size = max(self.config.fallback_size, self.config.min_component_area)
        if small_target_mode and expanded_aspect >= 1.85 and expanded_width >= expected_size * 2.35:
            return selection

        expanded_bbox = BoundingBox(
            x=patch.origin_x + work_bbox.x + x - self.config.padding,
            y=patch.origin_y + work_bbox.y + y - self.config.padding,
            width=expanded_width,
            height=expanded_height,
        )
        return SelectionResult(
            bbox=expanded_bbox,
            confidence=max(selection.confidence, min(1.0, area / max(self.config.min_component_area, 1))),
            polarity=selection.polarity,
        )

    def _select_contrast_component(
        self,
        patch: LocalPatch,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> SelectionResult | None:
        """Выбирает цель как отдельный контрастный компонент вокруг клика."""

        clicked_value = int(patch.image[patch.local_y, patch.local_x])
        patch_median = float(np.median(patch.image))
        hot_object = clicked_value >= patch_median
        polarity = "hot" if hot_object else "cold"

        blurred = cv2.GaussianBlur(patch.image, (5, 5), 0)
        threshold_mode = cv2.THRESH_BINARY if hot_object else cv2.THRESH_BINARY_INV
        _, object_mask = cv2.threshold(blurred, 0, 255, threshold_mode + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        bbox, area = self._component_bbox_for_click(object_mask, patch.local_x, patch.local_y)
        if bbox is None:
            return None

        split_bbox, split_area = self._split_contrast_cluster(
            blurred,
            object_mask,
            bbox,
            patch.local_x,
            patch.local_y,
            hot_object,
        )
        if split_bbox is not None:
            bbox = split_bbox
            area = split_area

        patch_area = patch.image.shape[0] * patch.image.shape[1]
        if area > int(patch_area * self.config.max_component_fill):
            return None
        if self._touches_local_patch_border(bbox, patch.image.shape):
            return None

        max_patch_width = int(patch.image.shape[1] * self.config.max_patch_span_ratio)
        max_patch_height = int(patch.image.shape[0] * self.config.max_patch_span_ratio)
        if bbox.width > max_patch_width or bbox.height > max_patch_height:
            return None

        component_mask = object_mask[bbox.y:bbox.y2, bbox.x:bbox.x2] > 0
        object_values = patch.image[bbox.y:bbox.y2, bbox.x:bbox.x2][component_mask]
        if object_values.size == 0:
            return None

        background_bbox = bbox.pad(self.config.background_ring, self.config.background_ring).clamp(patch.image.shape)
        background_mask = np.ones(
            (background_bbox.height, background_bbox.width),
            dtype=bool,
        )
        inner_x1 = bbox.x - background_bbox.x
        inner_y1 = bbox.y - background_bbox.y
        background_mask[
            inner_y1:inner_y1 + bbox.height,
            inner_x1:inner_x1 + bbox.width,
        ] = False
        background_values = patch.image[
            background_bbox.y:background_bbox.y2,
            background_bbox.x:background_bbox.x2,
        ][background_mask]
        if background_values.size < 10:
            return None

        object_level = float(np.median(object_values))
        background_level = float(np.median(background_values))
        contrast = object_level - background_level if hot_object else background_level - object_level
        if contrast < self.config.min_object_contrast:
            return None

        compact_selection = self._compact_selection_for_elongated_component(
            patch,
            bbox,
            area,
            patch.local_x,
            patch.local_y,
            hot_object,
            polarity,
            frame_shape,
        )
        if compact_selection is not None:
            return compact_selection

        result_bbox = BoundingBox(
            x=patch.origin_x + bbox.x - self.config.padding,
            y=patch.origin_y + bbox.y - self.config.padding,
            width=bbox.width + self.config.padding * 2,
            height=bbox.height + self.config.padding * 2,
        ).clamp(frame_shape)
        confidence = max(0.1, min(1.0, area / max(self.config.min_component_area, 1)))
        return SelectionResult(bbox=result_bbox, confidence=confidence, polarity=polarity)

    def _compact_selection_for_elongated_component(
        self,
        patch: LocalPatch,
        bbox: BoundingBox,
        area: int,
        click_x: int,
        click_y: int,
        hot_object: bool,
        polarity: str,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> SelectionResult | None:
        """Не даёт маленькой цели расползаться по длинной яркой линии."""

        if not self._component_is_oversized_for_small_target(bbox, area):
            return None

        radius = max(5, min(14, self.config.fallback_size))
        x1 = max(0, click_x - radius)
        y1 = max(0, click_y - radius)
        x2 = min(patch.image.shape[1], click_x + radius + 1)
        y2 = min(patch.image.shape[0], click_y + radius + 1)
        local = patch.image[y1:y2, x1:x2]
        if local.size == 0:
            return None

        blurred = cv2.GaussianBlur(local, (3, 3), 0)
        local_median = float(np.median(blurred))
        if hot_object:
            score = blurred.astype(np.float32) - local_median
        else:
            score = local_median - blurred.astype(np.float32)

        best_y, best_x = np.unravel_index(int(np.argmax(score)), score.shape)
        if float(score[best_y, best_x]) < self.config.min_object_contrast:
            center_x = click_x
            center_y = click_y
        else:
            center_x = x1 + int(best_x)
            center_y = y1 + int(best_y)

        size = max(self.config.fallback_size, self.config.min_component_area)
        compact_bbox = BoundingBox.from_center(
            patch.origin_x + center_x,
            patch.origin_y + center_y,
            size,
            size,
        ).clamp(frame_shape)
        return SelectionResult(bbox=compact_bbox, confidence=0.45, polarity=polarity)

    def _split_contrast_cluster(
        self,
        blurred_patch: np.ndarray,
        object_mask: np.ndarray,
        bbox: BoundingBox,
        click_x: int,
        click_y: int,
        hot_object: bool,
    ) -> tuple[BoundingBox | None, int]:
        """Разделяет крупный контрастный кластер на отдельные яркие ядра."""

        patch_area = object_mask.shape[0] * object_mask.shape[1]
        bbox_area = bbox.area
        if patch_area <= 0 or bbox_area <= 0:
            return None, 0
        if bbox_area < 1800 and max(bbox.width, bbox.height) < 55:
            return None, 0
        if bbox_area > patch_area * 0.25:
            return None, 0

        _, labels, _, _ = cv2.connectedComponentsWithStats(object_mask)
        clicked_label = int(labels[click_y, click_x])
        if clicked_label == 0:
            clicked_label = self._nearest_label(labels, click_x, click_y)
        if clicked_label == 0:
            return None, 0

        component_mask = labels == clicked_label
        component_values = blurred_patch[component_mask]
        if component_values.size < self.config.min_component_area * 4:
            return None, 0

        for quantile in (0.55, 0.65, 0.75):
            threshold = float(np.quantile(component_values, quantile if hot_object else 1.0 - quantile))
            if hot_object:
                core_mask = np.logical_and(component_mask, blurred_patch >= threshold)
            else:
                core_mask = np.logical_and(component_mask, blurred_patch <= threshold)

            core_mask_u8 = core_mask.astype(np.uint8) * 255
            _, core_labels, core_stats, core_centroids = cv2.connectedComponentsWithStats(core_mask_u8)
            candidates: list[tuple[int, float, int, int, BoundingBox]] = []
            for label in range(1, core_stats.shape[0]):
                area = int(core_stats[label, cv2.CC_STAT_AREA])
                if area < max(6, self.config.min_component_area // 2):
                    continue
                x = int(core_stats[label, cv2.CC_STAT_LEFT])
                y = int(core_stats[label, cv2.CC_STAT_TOP])
                w = int(core_stats[label, cv2.CC_STAT_WIDTH])
                h = int(core_stats[label, cv2.CC_STAT_HEIGHT])
                core_bbox = BoundingBox(x, y, w, h)
                contains_click = 0 if core_labels[click_y, click_x] == label else 1
                cx, cy = core_centroids[label]
                distance = float((cx - click_x) ** 2 + (cy - click_y) ** 2)
                candidates.append((contains_click, distance, -area, label, core_bbox))

            if len(candidates) < 2:
                continue

            _, _, _, selected_label, selected_core = min(candidates)
            selected_mask = (core_labels == selected_label).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            selected_mask = cv2.dilate(selected_mask, kernel, iterations=1)
            selected_mask = np.logical_and(selected_mask > 0, component_mask)
            points = np.column_stack(np.where(selected_mask))
            if points.size == 0:
                continue

            left = int(np.min(points[:, 1]))
            top = int(np.min(points[:, 0]))
            right = int(np.max(points[:, 1])) + 1
            bottom = int(np.max(points[:, 0])) + 1
            split_bbox = BoundingBox(left, top, right - left, bottom - top)
            split_area = int(np.count_nonzero(selected_mask))
            if split_bbox.area >= bbox_area * 0.78:
                continue
            return split_bbox, split_area

        return None, 0

    def _allowed_expansion_ratio(self, seed_area: int) -> float:
        """Ограничивает разрастание маленького ядра, но не душит крупные контрастные цели."""

        configured_ratio = float(self.config.max_expansion_ratio)
        if seed_area < 300:
            return min(configured_ratio, 3.0)
        if seed_area < 900:
            return min(configured_ratio, 3.2)
        if seed_area < 1600:
            return min(configured_ratio, 4.2)
        return configured_ratio

    def _allowed_expansion_side_ratio(self, seed_area: int) -> float:
        """Сдерживает расширение, когда рядом есть похожие отдельные объекты."""

        if seed_area < 300:
            return 1.8
        if seed_area < 900:
            return 1.95
        if seed_area < 1600:
            return 2.3
        return 3.0

    def _extract_patch(
        self,
        image: np.ndarray,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None,
        radius_override: int | None = None,
    ) -> LocalPatch:
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
        return LocalPatch(
            image=image[y1:y2, x1:x2],
            origin_x=x1,
            origin_y=y1,
            local_x=x - x1,
            local_y=y - y1,
        )

    def _build_mask(self, patch: np.ndarray, click_x: int, click_y: int) -> tuple[np.ndarray, str]:
        """Строит стартовую маску вокруг кликнутого пикселя."""

        return self._build_mask_with_tolerance_scale(patch, click_x, click_y, tolerance_scale=1.0)

    def _build_mask_with_tolerance_scale(
        self,
        patch: np.ndarray,
        click_x: int,
        click_y: int,
        *,
        tolerance_scale: float,
    ) -> tuple[np.ndarray, str]:
        """Строит маску с возможностью ужать допуск по яркости для крупной компоненты."""

        clicked_value = int(patch[click_y, click_x])
        median_value = float(np.median(patch))
        tolerance = self._estimate_tolerance(patch, click_x, click_y)
        tolerance = max(self.config.min_tolerance, int(round(tolerance * tolerance_scale)))

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
        patch: LocalPatch,
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
            tight_bbox, tight_area = self._tighten_large_component(
                patch,
                original_bbox=BoundingBox(x, y, w, h),
                original_area=area,
            )
            if tight_bbox is not None:
                x, y, w, h = tight_bbox.to_xywh()
                area = tight_area
            else:
                split_bbox = self._split_large_component(mask, labels, clicked_label, patch.local_x, patch.local_y)
                if split_bbox is not None:
                    x, y, w, h = split_bbox.to_xywh()

        max_patch_width = int(mask.shape[1] * self.config.max_patch_span_ratio)
        max_patch_height = int(mask.shape[0] * self.config.max_patch_span_ratio)
        if expected_bbox is None:
            if w > max_patch_width or h > max_patch_height:
                return None, 0.0
            compact_bbox = self._compact_local_bbox_for_large_component(patch, BoundingBox(x, y, w, h), area)
            if compact_bbox is not None:
                return compact_bbox, 0.45
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

    def _compact_local_bbox_for_large_component(
        self,
        patch: LocalPatch,
        bbox: BoundingBox,
        area: int,
    ) -> BoundingBox | None:
        """Возвращает компактный bbox, если компонент похож на длинную линию."""

        if not self._component_is_oversized_for_small_target(bbox, area):
            return None

        size = max(self.config.fallback_size, self.config.min_component_area)
        local_x = int(np.clip(patch.local_x - size // 2, 0, max(0, patch.image.shape[1] - size)))
        local_y = int(np.clip(patch.local_y - size // 2, 0, max(0, patch.image.shape[0] - size)))
        return BoundingBox(
            x=patch.origin_x + local_x,
            y=patch.origin_y + local_y,
            width=size,
            height=size,
        )

    def _component_is_oversized_for_small_target(self, bbox: BoundingBox, area: int) -> bool:
        """Проверяет, что компонент слишком большой для пресета маленькой цели."""

        small_target_mode = self.config.fallback_size <= 18 and self.config.max_expansion_ratio <= 2.5
        if not small_target_mode:
            return False

        expected_size = max(self.config.fallback_size, self.config.min_component_area)
        bbox_aspect = max(
            bbox.width / max(float(bbox.height), 1.0),
            bbox.height / max(float(bbox.width), 1.0),
        )
        oversized_side = bbox.width >= expected_size * 2.0 or bbox.height >= expected_size * 2.0
        oversized_area = bbox.area >= expected_size * expected_size * 4
        elongated = bbox_aspect >= 1.75
        long_line = bbox.width >= self.config.search_radius * 0.75

        return (oversized_side and (oversized_area or elongated)) or long_line or area >= max(expected_size * 24, 260)

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
        kernels = ((3, 3), (5, 5), (7, 7), (9, 9), (11, 11))

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
            if area < original_area * 0.08:
                continue
            if area < best_area * 0.88:
                best_bbox = bbox
                best_area = area

        return best_bbox

    def _tighten_large_component(
        self,
        patch: LocalPatch,
        original_bbox: BoundingBox,
        original_area: int,
    ) -> tuple[BoundingBox | None, int]:
        """Повторно выделяет слишком большую стартовую компоненту с более строгим порогом."""

        patch_area = patch.image.shape[0] * patch.image.shape[1]
        if patch_area <= 0:
            return None, 0

        area_fill = original_area / patch_area
        span_ratio = max(
            original_bbox.width / max(patch.image.shape[1], 1),
            original_bbox.height / max(patch.image.shape[0], 1),
        )
        if area_fill < 0.16 and span_ratio < 0.62:
            return None, 0

        original_bbox_area = max(original_bbox.area, 1)
        best_bbox: BoundingBox | None = None
        best_area = 0
        for tolerance_scale in (0.55, 0.4, 0.3):
            mask, _ = self._build_mask_with_tolerance_scale(
                patch.image,
                patch.local_x,
                patch.local_y,
                tolerance_scale=tolerance_scale,
            )
            bbox, area = self._component_bbox_for_click(mask, patch.local_x, patch.local_y)
            if bbox is None:
                continue
            if area < self.config.min_component_area * 3:
                continue
            if bbox.area > original_bbox_area * 0.78:
                continue
            best_bbox = bbox
            best_area = area
            break

        return best_bbox, best_area

    def _component_bbox_for_click(self, mask: np.ndarray, click_x: int, click_y: int) -> tuple[BoundingBox | None, int]:
        """Возвращает локальный bbox компоненты, ближайшей к клику."""

        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clicked_label = int(labels[click_y, click_x])
        if clicked_label == 0:
            clicked_label = self._nearest_label(labels, click_x, click_y)
            if clicked_label == 0:
                return None, 0

        area = int(stats[clicked_label, cv2.CC_STAT_AREA])
        if area < self.config.min_component_area:
            return None, 0

        bbox = BoundingBox(
            x=int(stats[clicked_label, cv2.CC_STAT_LEFT]),
            y=int(stats[clicked_label, cv2.CC_STAT_TOP]),
            width=int(stats[clicked_label, cv2.CC_STAT_WIDTH]),
            height=int(stats[clicked_label, cv2.CC_STAT_HEIGHT]),
        )
        return bbox, area

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
    def _touches_local_patch_border(
        bbox: BoundingBox,
        patch_shape: tuple[int, int] | tuple[int, int, int],
        tolerance: int = 2,
    ) -> bool:
        """Проверяет локальный bbox на касание края патча."""

        patch_h, patch_w = patch_shape[:2]
        return (
            bbox.x <= tolerance
            or bbox.y <= tolerance
            or bbox.x2 >= patch_w - tolerance
            or bbox.y2 >= patch_h - tolerance
        )

    @staticmethod
    def _touches_patch_border(bbox: BoundingBox, patch: LocalPatch, tolerance: int = 2) -> bool:
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
