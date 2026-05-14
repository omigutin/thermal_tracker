from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .....domain.models import BoundingBox
from ...result import TargetPolarity, TargetSelectorResult
from .contrast_component_mask_builder import ContrastComponentMaskBuilder
from .contrast_component_patch import ContrastComponentPatch


class ContrastComponentExtractorSettings(Protocol):
    """Описывает настройки, нужные извлекателю компоненты цели по клику."""

    # Отступ, добавляемый вокруг найденного bbox цели.
    padding: int
    # Минимальная площадь компоненты, которую можно считать целью.
    min_component_area: int
    # Максимальная доля площади patch, которую может занимать компонента.
    max_component_fill: float
    # Максимальная доля ширины/высоты patch, которую может занимать bbox компоненты.
    max_patch_span_ratio: float
    # Максимальный рост bbox при уточнении уже выбранной цели.
    max_refine_growth: float
    # Размер fallback-bbox для маленькой цели.
    fallback_size: int
    # Максимальный коэффициент расширения маленького ядра цели.
    max_expansion_ratio: float
    # Радиус поиска вокруг клика.
    search_radius: int
    # Толщина фонового кольца вокруг компоненты для оценки контраста.
    background_ring: int
    # Минимальный контраст объекта относительно окружения.
    min_object_contrast: float

    # Размер ядра размытия для контрастной компоненты.
    contrast_blur_kernel: int
    # Размер ядра морфологической очистки контрастной маски.
    contrast_morphology_kernel: int
    # Количество итераций удаления шума из контрастной маски.
    contrast_open_iterations: int
    # Количество итераций закрытия разрывов в контрастной маске.
    contrast_close_iterations: int
    # Минимальное количество фоновых пикселей для оценки контраста.
    min_background_pixels: int
    # Уверенность для компактного fallback-выбора.
    compact_selection_confidence: float
    # Минимальный радиус компактного выбора вокруг клика.
    compact_min_radius: int
    # Максимальный радиус компактного выбора вокруг клика.
    compact_max_radius: int

    # Минимальная площадь bbox, с которой можно пытаться делить контрастный кластер.
    contrast_split_min_bbox_area: int
    # Минимальная сторона bbox, с которой можно пытаться делить контрастный кластер.
    contrast_split_min_bbox_side: int
    # Максимальная доля patch, которую может занимать bbox для деления контрастного кластера.
    contrast_split_max_patch_fill: float
    # Во сколько раз компонент должен быть больше минимальной площади для деления.
    contrast_split_min_area_multiplier: int
    # Квантили, которыми пробуем выделять ядра внутри контрастного кластера.
    contrast_split_quantiles: tuple[float, ...]
    # Минимальная площадь ядра при делении контрастного кластера.
    contrast_core_min_area: int
    # Доля min_component_area для минимальной площади ядра.
    contrast_core_min_area_ratio: float
    # Размер ядра расширения выбранного ядра контрастного кластера.
    contrast_core_dilate_kernel: int
    # Количество итераций расширения выбранного ядра контрастного кластера.
    contrast_core_dilate_iterations: int
    # Максимальная доля исходного bbox, которую может занимать split-bbox.
    contrast_split_max_bbox_ratio: float

    # Во сколько раз крупная компонента должна превышать min_component_area для деления.
    large_component_min_area_multiplier: int
    # Размеры ядер эрозии для попыток разделить крупную компоненту.
    split_kernel_sizes: tuple[int, ...]
    # Минимальная доля исходной площади для принятия split-кандидата.
    split_min_area_ratio: float
    # Требуемое улучшение площади split-кандидата относительно текущего лучшего.
    split_improvement_ratio: float

    # Минимальная заполненность patch для повторного ужесточения крупной компоненты.
    tighten_min_area_fill: float
    # Минимальная доля span компоненты для повторного ужесточения.
    tighten_min_span_ratio: float
    # Масштабы tolerance для повторного ужесточения крупной компоненты.
    tighten_tolerance_scales: tuple[float, ...]
    # Минимальный множитель площади при повторном ужесточении.
    tighten_min_area_multiplier: int
    # Максимальная доля исходного bbox для принятия ужатой компоненты.
    tighten_max_bbox_area_ratio: float

    # Максимальный fallback_size, при котором включается режим маленькой цели.
    small_target_max_fallback_size: int
    # Максимальный max_expansion_ratio, при котором включается режим маленькой цели.
    small_target_max_expansion_ratio: float
    # Множитель стороны ожидаемой маленькой цели для признака oversized.
    oversized_side_multiplier: float
    # Множитель площади ожидаемой маленькой цели для признака oversized.
    oversized_area_multiplier: float
    # Aspect ratio, после которого компонент считается вытянутым.
    elongated_aspect_ratio: float
    # Доля search_radius, после которой длинный компонент считается линией.
    long_line_search_radius_ratio: float
    # Множитель площади ожидаемой цели для отсечения огромных компонент.
    oversized_area_min_multiplier: int
    # Минимальная абсолютная площадь для отсечения огромных компонент.
    oversized_area_min_pixels: int


@dataclass(slots=True)
class ContrastComponentExtractor:
    """Извлекает bbox цели из компоненты связности около точки клика."""

    config: ContrastComponentExtractorSettings
    mask_builder: ContrastComponentMaskBuilder

    def extract_component(
        self,
        mask: np.ndarray,
        patch: ContrastComponentPatch,
        expected_bbox: BoundingBox | None,
    ) -> tuple[BoundingBox | None, float]:
        """Вернуть bbox компоненты, связанной с точкой клика.

        Если клик попал в дырку маски, берётся ближайшая непустая компонента.
        Для нового выбора дополнительно выполняются попытки ужать или разделить
        слишком крупную компоненту.
        """
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clicked_label = int(labels[patch.local_y, patch.local_x])

        if clicked_label == 0:
            clicked_label = self.nearest_label(
                labels=labels,
                x=patch.local_x,
                y=patch.local_y,
            )
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
        width = int(stats[clicked_label, cv2.CC_STAT_WIDTH])
        height = int(stats[clicked_label, cv2.CC_STAT_HEIGHT])

        if expected_bbox is None:
            bbox = BoundingBox(x=x, y=y, width=width, height=height)
            tightened_bbox, tightened_area = self.tighten_large_component(
                patch=patch,
                original_bbox=bbox,
                original_area=area,
            )

            if tightened_bbox is not None:
                x, y, width, height = tightened_bbox.to_xywh()
                area = tightened_area
            else:
                split_bbox = self.split_large_component(
                    mask=mask,
                    labels=labels,
                    clicked_label=clicked_label,
                    click_x=patch.local_x,
                    click_y=patch.local_y,
                )
                if split_bbox is not None:
                    x, y, width, height = split_bbox.to_xywh()

        if expected_bbox is None:
            if self._exceeds_patch_span(
                width=width,
                height=height,
                patch_shape=mask.shape,
            ):
                return None, 0.0

            compact_bbox = self.compact_local_bbox_for_large_component(
                patch=patch,
                bbox=BoundingBox(x=x, y=y, width=width, height=height),
                area=area,
            )
            if compact_bbox is not None:
                return compact_bbox, self.config.compact_selection_confidence
        else:
            if width > int(expected_bbox.width * self.config.max_refine_growth):
                return None, 0.0
            if height > int(expected_bbox.height * self.config.max_refine_growth):
                return None, 0.0

        bbox = BoundingBox(
            x=patch.origin_x + x - self.config.padding,
            y=patch.origin_y + y - self.config.padding,
            width=width + self.config.padding * 2,
            height=height + self.config.padding * 2,
        )

        confidence = max(
            0.1,
            min(1.0, area / max(self.config.min_component_area, 1)),
        )
        return bbox, confidence

    def select_contrast_component(
        self,
        patch: ContrastComponentPatch,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> TargetSelectorResult | None:
        """Выбрать цель как отдельный контрастный компонент вокруг клика.

        Этот путь нужен для первого клика: он пытается найти яркое или холодное
        ядро цели по локальному контрасту до обычной похожести по яркости.
        """
        clicked_value = int(patch.image[patch.local_y, patch.local_x])
        patch_median = float(np.median(patch.image))
        hot_object = clicked_value >= patch_median
        polarity = TargetPolarity.HOT if hot_object else TargetPolarity.COLD

        blurred = cv2.GaussianBlur(
            patch.image,
            (self.config.contrast_blur_kernel, self.config.contrast_blur_kernel),
            0,
        )
        threshold_mode = cv2.THRESH_BINARY if hot_object else cv2.THRESH_BINARY_INV
        _, object_mask = cv2.threshold(
            blurred,
            0,
            255,
            threshold_mode + cv2.THRESH_OTSU,
        )
        object_mask = self._clean_contrast_mask(object_mask)

        bbox, area = self.component_bbox_for_click(
            mask=object_mask,
            click_x=patch.local_x,
            click_y=patch.local_y,
        )
        if bbox is None:
            return None

        split_bbox, split_area = self.split_contrast_cluster(
            blurred_patch=blurred,
            object_mask=object_mask,
            bbox=bbox,
            click_x=patch.local_x,
            click_y=patch.local_y,
            hot_object=hot_object,
        )
        if split_bbox is not None:
            bbox = split_bbox
            area = split_area

        if not self._accept_contrast_component(
            patch=patch,
            object_mask=object_mask,
            bbox=bbox,
            area=area,
        ):
            return None

        compact_selection = self.compact_selection_for_elongated_component(
            patch=patch,
            bbox=bbox,
            area=area,
            click_x=patch.local_x,
            click_y=patch.local_y,
            hot_object=hot_object,
            polarity=polarity,
            frame_shape=frame_shape,
        )
        if compact_selection is not None:
            return compact_selection

        result_bbox = BoundingBox(
            x=patch.origin_x + bbox.x - self.config.padding,
            y=patch.origin_y + bbox.y - self.config.padding,
            width=bbox.width + self.config.padding * 2,
            height=bbox.height + self.config.padding * 2,
        ).clamp(frame_shape)

        confidence = max(
            0.1,
            min(1.0, area / max(self.config.min_component_area, 1)),
        )
        return TargetSelectorResult(
            bbox=result_bbox,
            confidence=confidence,
            polarity=polarity,
        )

    def compact_selection_for_elongated_component(
        self,
        patch: ContrastComponentPatch,
        bbox: BoundingBox,
        area: int,
        click_x: int,
        click_y: int,
        hot_object: bool,
        polarity: TargetPolarity,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> TargetSelectorResult | None:
        """Вернуть компактный выбор, если компонент похож на длинную линию."""
        if not self.component_is_oversized_for_small_target(bbox=bbox, area=area):
            return None

        radius = max(
            self.config.compact_min_radius,
            min(self.config.compact_max_radius, self.config.fallback_size),
        )
        x1 = max(0, click_x - radius)
        y1 = max(0, click_y - radius)
        x2 = min(patch.width, click_x + radius + 1)
        y2 = min(patch.height, click_y + radius + 1)

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

        return TargetSelectorResult(
            bbox=compact_bbox,
            confidence=self.config.compact_selection_confidence,
            polarity=polarity,
        )

    def split_contrast_cluster(
        self,
        blurred_patch: np.ndarray,
        object_mask: np.ndarray,
        bbox: BoundingBox,
        click_x: int,
        click_y: int,
        hot_object: bool,
    ) -> tuple[BoundingBox | None, int]:
        """Разделить крупный контрастный кластер на отдельные ядра."""
        patch_area = object_mask.shape[0] * object_mask.shape[1]
        bbox_area = bbox.area

        if patch_area <= 0 or bbox_area <= 0:
            return None, 0
        if (
            bbox_area < self.config.contrast_split_min_bbox_area
            and max(bbox.width, bbox.height) < self.config.contrast_split_min_bbox_side
        ):
            return None, 0
        if bbox_area > patch_area * self.config.contrast_split_max_patch_fill:
            return None, 0

        _, labels, _, _ = cv2.connectedComponentsWithStats(object_mask)
        clicked_label = int(labels[click_y, click_x])

        if clicked_label == 0:
            clicked_label = self.nearest_label(labels=labels, x=click_x, y=click_y)
        if clicked_label == 0:
            return None, 0

        component_mask = labels == clicked_label
        component_values = blurred_patch[component_mask]

        if (
            component_values.size
            < self.config.min_component_area
            * self.config.contrast_split_min_area_multiplier
        ):
            return None, 0

        for quantile in self.config.contrast_split_quantiles:
            threshold = float(
                np.quantile(
                    component_values,
                    quantile if hot_object else 1.0 - quantile,
                )
            )
            if hot_object:
                core_mask = np.logical_and(component_mask, blurred_patch >= threshold)
            else:
                core_mask = np.logical_and(component_mask, blurred_patch <= threshold)

            split_bbox, split_area = self._select_split_core(
                core_mask=core_mask,
                component_mask=component_mask,
                click_x=click_x,
                click_y=click_y,
                bbox_area=bbox_area,
            )
            if split_bbox is not None:
                return split_bbox, split_area

        return None, 0

    def split_large_component(
        self,
        mask: np.ndarray,
        labels: np.ndarray,
        clicked_label: int,
        click_x: int,
        click_y: int,
    ) -> BoundingBox | None:
        """Попытаться расколоть слишком крупную компоненту на отдельные объекты."""
        component_mask = (labels == clicked_label).astype(np.uint8) * 255
        original_area = int(np.count_nonzero(component_mask))

        if (
            original_area
            < self.config.min_component_area
            * self.config.large_component_min_area_multiplier
        ):
            return None

        best_bbox: BoundingBox | None = None
        best_area = original_area

        for kernel_size in self.config.split_kernel_sizes:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            eroded = cv2.erode(component_mask, kernel, iterations=1)

            if int(np.count_nonzero(eroded)) < self.config.min_component_area:
                continue

            candidate = self._select_split_candidate(
                eroded=eroded,
                kernel=kernel,
                click_x=click_x,
                click_y=click_y,
            )
            if candidate is None:
                continue

            bbox, area = candidate

            if area < original_area * self.config.split_min_area_ratio:
                continue
            if area < best_area * self.config.split_improvement_ratio:
                best_bbox = bbox
                best_area = area

        return best_bbox

    def tighten_large_component(
        self,
        patch: ContrastComponentPatch,
        original_bbox: BoundingBox,
        original_area: int,
    ) -> tuple[BoundingBox | None, int]:
        """Повторно выделить слишком большую компоненту с более строгим tolerance."""
        patch_area = patch.height * patch.width
        if patch_area <= 0:
            return None, 0

        area_fill = original_area / patch_area
        span_ratio = max(
            original_bbox.width / max(patch.width, 1),
            original_bbox.height / max(patch.height, 1),
        )

        if (
            area_fill < self.config.tighten_min_area_fill
            and span_ratio < self.config.tighten_min_span_ratio
        ):
            return None, 0

        original_bbox_area = max(original_bbox.area, 1)

        for tolerance_scale in self.config.tighten_tolerance_scales:
            mask, _ = self.mask_builder.build_mask_with_tolerance_scale(
                patch=patch.image,
                click_x=patch.local_x,
                click_y=patch.local_y,
                tolerance_scale=tolerance_scale,
            )
            bbox, area = self.component_bbox_for_click(
                mask=mask,
                click_x=patch.local_x,
                click_y=patch.local_y,
            )

            if bbox is None:
                continue
            if (
                area
                < self.config.min_component_area
                * self.config.tighten_min_area_multiplier
            ):
                continue
            if bbox.area > original_bbox_area * self.config.tighten_max_bbox_area_ratio:
                continue

            return bbox, area

        return None, 0

    def component_bbox_for_click(
        self,
        mask: np.ndarray,
        click_x: int,
        click_y: int,
    ) -> tuple[BoundingBox | None, int]:
        """Вернуть локальный bbox компоненты, ближайшей к клику."""
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clicked_label = int(labels[click_y, click_x])

        if clicked_label == 0:
            clicked_label = self.nearest_label(labels=labels, x=click_x, y=click_y)
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

    def compact_local_bbox_for_large_component(
        self,
        patch: ContrastComponentPatch,
        bbox: BoundingBox,
        area: int,
    ) -> BoundingBox | None:
        """Вернуть компактный bbox, если компонент слишком велик для маленькой цели."""
        if not self.component_is_oversized_for_small_target(bbox=bbox, area=area):
            return None

        size = max(self.config.fallback_size, self.config.min_component_area)
        local_x = int(np.clip(patch.local_x - size // 2, 0, max(0, patch.width - size)))
        local_y = int(np.clip(patch.local_y - size // 2, 0, max(0, patch.height - size)))

        return BoundingBox(
            x=patch.origin_x + local_x,
            y=patch.origin_y + local_y,
            width=size,
            height=size,
        )

    def component_is_oversized_for_small_target(
        self,
        bbox: BoundingBox,
        area: int,
    ) -> bool:
        """Проверить, что компонент слишком большой для режима маленькой цели."""
        small_target_mode = (
            self.config.fallback_size <= self.config.small_target_max_fallback_size
            and self.config.max_expansion_ratio
            <= self.config.small_target_max_expansion_ratio
        )
        if not small_target_mode:
            return False

        expected_size = max(self.config.fallback_size, self.config.min_component_area)
        bbox_aspect = max(
            bbox.width / max(float(bbox.height), 1.0),
            bbox.height / max(float(bbox.width), 1.0),
        )

        oversized_side = (
            bbox.width >= expected_size * self.config.oversized_side_multiplier
            or bbox.height >= expected_size * self.config.oversized_side_multiplier
        )
        oversized_area = (
            bbox.area
            >= expected_size * expected_size * self.config.oversized_area_multiplier
        )
        elongated = bbox_aspect >= self.config.elongated_aspect_ratio
        long_line = (
            bbox.width
            >= self.config.search_radius * self.config.long_line_search_radius_ratio
        )
        huge_area = area >= max(
            expected_size * self.config.oversized_area_min_multiplier,
            self.config.oversized_area_min_pixels,
        )

        return (
            (oversized_side and (oversized_area or elongated))
            or long_line
            or huge_area
        )

    @staticmethod
    def touches_local_patch_border(
        bbox: BoundingBox,
        patch_shape: tuple[int, int] | tuple[int, int, int],
        tolerance: int = 2,
    ) -> bool:
        """Проверить локальный bbox на касание края patch."""
        patch_height, patch_width = patch_shape[:2]

        return (
            bbox.x <= tolerance
            or bbox.y <= tolerance
            or bbox.x2 >= patch_width - tolerance
            or bbox.y2 >= patch_height - tolerance
        )

    @staticmethod
    def nearest_label(labels: np.ndarray, x: int, y: int) -> int:
        """Найти ближайшую непустую компоненту, если клик попал в дырку."""
        window = labels[max(0, y - 2): y + 3, max(0, x - 2): x + 3]
        non_zero = window[window > 0]

        if non_zero.size == 0:
            return 0

        values, counts = np.unique(non_zero, return_counts=True)
        return int(values[np.argmax(counts)])

    def _accept_contrast_component(
        self,
        patch: ContrastComponentPatch,
        object_mask: np.ndarray,
        bbox: BoundingBox,
        area: int,
    ) -> bool:
        """Проверить, можно ли принять контрастную компоненту как цель."""
        patch_area = patch.height * patch.width

        if area > int(patch_area * self.config.max_component_fill):
            return False
        if self.touches_local_patch_border(bbox=bbox, patch_shape=patch.shape):
            return False
        if self._exceeds_patch_span(
            width=bbox.width,
            height=bbox.height,
            patch_shape=patch.shape,
        ):
            return False

        component_mask = object_mask[bbox.y:bbox.y2, bbox.x:bbox.x2] > 0
        object_values = patch.image[bbox.y:bbox.y2, bbox.x:bbox.x2][component_mask]

        if object_values.size == 0:
            return False

        background_bbox = bbox.pad(
            self.config.background_ring,
            self.config.background_ring,
        ).clamp(patch.shape)
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

        if background_values.size < self.config.min_background_pixels:
            return False

        object_level = float(np.median(object_values))
        background_level = float(np.median(background_values))
        contrast = abs(object_level - background_level)

        return contrast >= self.config.min_object_contrast

    def _clean_contrast_mask(self, mask: np.ndarray) -> np.ndarray:
        """Очистить контрастную бинарную маску."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.config.contrast_morphology_kernel,
                self.config.contrast_morphology_kernel,
            ),
        )

        if self.config.contrast_open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self.config.contrast_open_iterations,
            )

        if self.config.contrast_close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.config.contrast_close_iterations,
            )

        return mask

    def _select_split_core(
        self,
        core_mask: np.ndarray,
        component_mask: np.ndarray,
        click_x: int,
        click_y: int,
        bbox_area: int,
    ) -> tuple[BoundingBox | None, int]:
        """Выбрать ядро компоненты, наиболее связанное с кликом."""
        core_mask_u8 = core_mask.astype(np.uint8) * 255
        _, core_labels, core_stats, core_centroids = cv2.connectedComponentsWithStats(
            core_mask_u8,
        )

        candidates: list[tuple[int, float, int, int, BoundingBox]] = []

        min_core_area = max(
            self.config.contrast_core_min_area,
            int(self.config.min_component_area * self.config.contrast_core_min_area_ratio),
        )

        for label in range(1, core_stats.shape[0]):
            area = int(core_stats[label, cv2.CC_STAT_AREA])
            if area < min_core_area:
                continue

            x = int(core_stats[label, cv2.CC_STAT_LEFT])
            y = int(core_stats[label, cv2.CC_STAT_TOP])
            width = int(core_stats[label, cv2.CC_STAT_WIDTH])
            height = int(core_stats[label, cv2.CC_STAT_HEIGHT])
            core_bbox = BoundingBox(x=x, y=y, width=width, height=height)

            contains_click = 0 if core_labels[click_y, click_x] == label else 1
            center_x, center_y = core_centroids[label]
            distance = float((center_x - click_x) ** 2 + (center_y - click_y) ** 2)

            candidates.append((contains_click, distance, -area, label, core_bbox))

        if len(candidates) < 2:
            return None, 0

        _, _, _, selected_label, _ = min(candidates)
        selected_mask = (core_labels == selected_label).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.config.contrast_core_dilate_kernel,
                self.config.contrast_core_dilate_kernel,
            ),
        )
        selected_mask = cv2.dilate(
            selected_mask,
            kernel,
            iterations=self.config.contrast_core_dilate_iterations,
        )
        selected_mask = np.logical_and(selected_mask > 0, component_mask)

        points = np.column_stack(np.where(selected_mask))
        if points.size == 0:
            return None, 0

        left = int(np.min(points[:, 1]))
        top = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 1])) + 1
        bottom = int(np.max(points[:, 0])) + 1

        split_bbox = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
        split_area = int(np.count_nonzero(selected_mask))

        if split_bbox.area >= bbox_area * self.config.contrast_split_max_bbox_ratio:
            return None, 0

        return split_bbox, split_area

    def _select_split_candidate(
        self,
        eroded: np.ndarray,
        kernel: np.ndarray,
        click_x: int,
        click_y: int,
    ) -> tuple[BoundingBox, int] | None:
        """Выбрать лучший split-кандидат после эрозии крупной компоненты."""
        _, split_labels, split_stats, _ = cv2.connectedComponentsWithStats(eroded)

        candidate_labels = [
            label
            for label in range(1, split_stats.shape[0])
            if int(split_stats[label, cv2.CC_STAT_AREA])
            >= self.config.min_component_area
        ]

        if len(candidate_labels) < 2:
            return None

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

            bbox = BoundingBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
            area = int(np.count_nonzero(restored))

            if restored[click_y, click_x] > 0:
                distance_score = -1.0
            else:
                center_x, center_y = bbox.center
                distance_score = float(
                    (center_x - click_x) ** 2 + (center_y - click_y) ** 2
                )

            candidate = (distance_score, bbox, area)

            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        if best_candidate is None:
            return None

        _, bbox, area = best_candidate
        return bbox, area

    def _exceeds_patch_span(
        self,
        width: int,
        height: int,
        patch_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        """Проверить, что bbox занимает слишком большую часть patch."""
        patch_height, patch_width = patch_shape[:2]
        max_patch_width = int(patch_width * self.config.max_patch_span_ratio)
        max_patch_height = int(patch_height * self.config.max_patch_span_ratio)

        return width > max_patch_width or height > max_patch_height