from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .....domain.models import BoundingBox
from ...result import TargetPolarity, TargetSelectorResult
from .contrast_component_patch import ContrastComponentPatch


class ClickSelectionExpanderSettings(Protocol):
    """Описывает настройки, нужные расширителю выбранной области."""

    # Отступ, добавляемый вокруг найденной области.
    padding: int
    # Минимальная площадь компоненты, используемая для оценки уверенности.
    min_component_area: int
    # Размер fallback-bbox для маленькой цели.
    fallback_size: int
    # Максимальный коэффициент расширения площади найденной области.
    max_expansion_ratio: float

    # Минимальный внешний отступ вокруг seed-области для поиска полного объекта.
    expansion_margin: int
    # Множитель размера seed-области для расчёта рабочего отступа.
    expansion_margin_seed_scale: float
    # Максимальная площадь seed-области, которую имеет смысл расширять.
    expansion_seed_max_area: int
    # Максимальная доля рабочего patch, которую может занять расширенная область.
    max_expanded_fill: float
    # Доля контраста между объектом и фоном для построения порога foreground.
    foreground_fraction: float
    # Толщина фонового кольца вокруг seed-области.
    background_ring: int
    # Минимальный контраст объекта относительно фона.
    min_object_contrast: float
    # Минимальное количество фоновых пикселей для оценки контраста.
    min_background_pixels: int

    # Размер ядра морфологической очистки расширенной маски.
    expansion_morphology_kernel: int
    # Количество итераций удаления мелкого шума из расширенной маски.
    expansion_open_iterations: int
    # Количество итераций закрытия разрывов в расширенной маске.
    expansion_close_iterations: int
    # Количество итераций расширения seed-маски для поиска пересечённой компоненты.
    expansion_seed_dilate_iterations: int

    # Верхняя граница площади seed, для которой expansion ratio сильно ограничен.
    expansion_small_area_limit: int
    # Верхняя граница площади seed для среднего ограничения expansion ratio.
    expansion_medium_area_limit: int
    # Верхняя граница площади seed для мягкого ограничения expansion ratio.
    expansion_large_area_limit: int
    # Максимальный expansion ratio для маленького seed.
    expansion_small_area_ratio: float
    # Максимальный expansion ratio для среднего seed.
    expansion_medium_area_ratio: float
    # Максимальный expansion ratio для крупного seed.
    expansion_large_area_ratio: float

    # Максимальный рост стороны bbox для маленького seed.
    expansion_small_side_ratio: float
    # Максимальный рост стороны bbox для среднего seed.
    expansion_medium_side_ratio: float
    # Максимальный рост стороны bbox для крупного seed.
    expansion_large_side_ratio: float
    # Максимальный рост стороны bbox для остальных seed.
    expansion_default_side_ratio: float

    # Максимальный fallback_size, при котором включается режим маленькой цели.
    small_target_max_fallback_size: int
    # Максимальный max_expansion_ratio, при котором включается режим маленькой цели.
    small_target_max_expansion_ratio: float
    # Aspect ratio, после которого расширение маленькой цели считается подозрительным.
    small_target_expanded_aspect_limit: float
    # Множитель ожидаемого размера, после которого ширина расширения считается подозрительной.
    small_target_expanded_width_multiplier: float


@dataclass(slots=True)
class ClickSelectionExpander:
    """Расширяет выбранное контрастное ядро до bbox всего объекта."""

    config: ClickSelectionExpanderSettings

    def expand_selection(
        self,
        patch: ContrastComponentPatch,
        selection: TargetSelectorResult,
    ) -> TargetSelectorResult:
        """Попытаться расширить выбранное ядро до полной области цели.

        Сначала seed переводится в координаты локального patch. Затем вокруг
        него берётся рабочая область, оценивается фон и объект, строится маска
        foreground и выбирается компонента, пересекающая seed.
        """
        local_seed = patch.to_local_bbox(selection.bbox).clamp(patch.shape)

        if local_seed.area <= 0:
            return selection
        if local_seed.area > self.config.expansion_seed_max_area:
            return selection

        work_bbox = self._build_work_bbox(
            seed_bbox=local_seed,
            patch_shape=patch.shape,
        )
        work_patch = patch.image[work_bbox.y:work_bbox.y2, work_bbox.x:work_bbox.x2]

        if work_patch.size == 0:
            return selection

        seed_mask = self._build_seed_mask(
            seed_bbox=local_seed,
            work_bbox=work_bbox,
            work_shape=work_patch.shape,
        )
        foreground_mask = self._build_foreground_mask(
            work_patch=work_patch,
            seed_mask=seed_mask,
            polarity=selection.polarity,
        )

        if foreground_mask is None:
            return selection

        expanded = self._select_expanded_component(
            foreground_mask=foreground_mask,
            seed_mask=seed_mask,
            work_bbox=work_bbox,
            patch=patch,
            selection=selection,
        )

        if expanded is None:
            return selection

        expanded_bbox, expanded_area = expanded

        if not self._accept_expanded_bbox(
            expanded_bbox=expanded_bbox,
            expanded_area=expanded_area,
            selection=selection,
        ):
            return selection

        return TargetSelectorResult(
            bbox=expanded_bbox,
            confidence=max(
                selection.confidence,
                min(1.0, expanded_area / max(self.config.min_component_area, 1)),
            ),
            polarity=selection.polarity,
        )

    def touches_patch_border(
        self,
        bbox: BoundingBox,
        patch: ContrastComponentPatch,
        tolerance: int = 2,
    ) -> bool:
        """Проверить, упёрлась ли найденная область в край локального patch."""
        local_bbox = patch.to_local_bbox(bbox)

        return (
            local_bbox.x <= tolerance
            or local_bbox.y <= tolerance
            or local_bbox.x2 >= patch.width - tolerance
            or local_bbox.y2 >= patch.height - tolerance
        )

    def allowed_expansion_ratio(self, seed_area: int) -> float:
        """Вернуть допустимый рост площади для seed-области."""
        configured_ratio = float(self.config.max_expansion_ratio)

        if seed_area < self.config.expansion_small_area_limit:
            return min(configured_ratio, self.config.expansion_small_area_ratio)
        if seed_area < self.config.expansion_medium_area_limit:
            return min(configured_ratio, self.config.expansion_medium_area_ratio)
        if seed_area < self.config.expansion_large_area_limit:
            return min(configured_ratio, self.config.expansion_large_area_ratio)

        return configured_ratio

    def allowed_expansion_side_ratio(self, seed_area: int) -> float:
        """Вернуть допустимый рост стороны bbox для seed-области."""
        if seed_area < self.config.expansion_small_area_limit:
            return self.config.expansion_small_side_ratio
        if seed_area < self.config.expansion_medium_area_limit:
            return self.config.expansion_medium_side_ratio
        if seed_area < self.config.expansion_large_area_limit:
            return self.config.expansion_large_side_ratio

        return self.config.expansion_default_side_ratio

    def _build_work_bbox(
        self,
        seed_bbox: BoundingBox,
        patch_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Построить рабочую область вокруг seed."""
        margin = max(
            self.config.expansion_margin,
            int(
                round(
                    max(seed_bbox.width, seed_bbox.height)
                    * self.config.expansion_margin_seed_scale
                )
            ),
        )

        return seed_bbox.pad(margin, margin).clamp(patch_shape)

    def _build_seed_mask(
        self,
        seed_bbox: BoundingBox,
        work_bbox: BoundingBox,
        work_shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """Построить seed-маску внутри рабочей области."""
        seed_x1 = seed_bbox.x - work_bbox.x
        seed_y1 = seed_bbox.y - work_bbox.y
        seed_x2 = seed_x1 + seed_bbox.width
        seed_y2 = seed_y1 + seed_bbox.height

        seed_mask = np.zeros(work_shape[:2], dtype=np.uint8)
        seed_mask[seed_y1:seed_y2, seed_x1:seed_x2] = 255

        return seed_mask

    def _build_foreground_mask(
        self,
        work_patch: np.ndarray,
        seed_mask: np.ndarray,
        polarity: TargetPolarity,
    ) -> np.ndarray | None:
        """Построить маску объекта по контрасту seed относительно фона."""
        object_values = work_patch[seed_mask > 0]
        if object_values.size == 0:
            return None

        background_mask = self._build_background_mask(
            seed_mask=seed_mask,
            seed_bbox=self._bbox_from_seed_mask(seed_mask),
        )
        background_values = work_patch[background_mask]

        if background_values.size < self.config.min_background_pixels:
            return None

        object_level = float(np.median(object_values))
        background_level = float(np.median(background_values))
        contrast = object_level - background_level

        if abs(contrast) < self.config.min_object_contrast:
            return None

        threshold = background_level + contrast * self.config.foreground_fraction

        if polarity == TargetPolarity.COLD:
            foreground_mask = (
                work_patch.astype(np.float32) <= threshold
            ).astype(np.uint8) * 255
        else:
            foreground_mask = (
                work_patch.astype(np.float32) >= threshold
            ).astype(np.uint8) * 255

        return self._clean_foreground_mask(foreground_mask)

    def _build_background_mask(
        self,
        seed_mask: np.ndarray,
        seed_bbox: BoundingBox,
    ) -> np.ndarray:
        """Построить маску фонового кольца вокруг seed."""
        background_mask = np.ones(seed_mask.shape, dtype=bool)
        background_inner = seed_bbox.pad(
            self.config.background_ring,
            self.config.background_ring,
        ).clamp(seed_mask.shape)

        background_mask[
            background_inner.y:background_inner.y2,
            background_inner.x:background_inner.x2,
        ] = False

        return background_mask

    def _select_expanded_component(
        self,
        foreground_mask: np.ndarray,
        seed_mask: np.ndarray,
        work_bbox: BoundingBox,
        patch: ContrastComponentPatch,
        selection: TargetSelectorResult,
    ) -> tuple[BoundingBox, int] | None:
        """Выбрать foreground-компоненту, пересекающую seed."""
        kernel = self._build_morphology_kernel()
        seed_support = cv2.dilate(
            seed_mask,
            kernel,
            iterations=self.config.expansion_seed_dilate_iterations,
        )

        _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask)
        overlapping_labels = np.unique(labels[seed_support > 0])
        overlapping_labels = overlapping_labels[overlapping_labels > 0]

        if overlapping_labels.size == 0:
            return None

        best_label = max(
            overlapping_labels.tolist(),
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
        area = int(stats[best_label, cv2.CC_STAT_AREA])

        if area <= selection.bbox.area:
            return None

        x = int(stats[best_label, cv2.CC_STAT_LEFT])
        y = int(stats[best_label, cv2.CC_STAT_TOP])
        width = int(stats[best_label, cv2.CC_STAT_WIDTH])
        height = int(stats[best_label, cv2.CC_STAT_HEIGHT])

        expanded_bbox = BoundingBox(
            x=patch.origin_x + work_bbox.x + x - self.config.padding,
            y=patch.origin_y + work_bbox.y + y - self.config.padding,
            width=width + self.config.padding * 2,
            height=height + self.config.padding * 2,
        )

        return expanded_bbox, area

    def _accept_expanded_bbox(
        self,
        expanded_bbox: BoundingBox,
        expanded_area: int,
        selection: TargetSelectorResult,
    ) -> bool:
        """Проверить, не слишком ли агрессивно расширилась область."""
        if expanded_area > int(selection.bbox.area * self.allowed_expansion_ratio(selection.bbox.area)):
            return False

        allowed_side_ratio = self.allowed_expansion_side_ratio(selection.bbox.area)

        if expanded_bbox.width > int(selection.bbox.width * allowed_side_ratio):
            return False
        if expanded_bbox.height > int(selection.bbox.height * allowed_side_ratio):
            return False

        expanded_aspect = max(
            expanded_bbox.width / max(float(expanded_bbox.height), 1.0),
            expanded_bbox.height / max(float(expanded_bbox.width), 1.0),
        )

        small_target_mode = (
            self.config.fallback_size <= self.config.small_target_max_fallback_size
            and self.config.max_expansion_ratio
            <= self.config.small_target_max_expansion_ratio
        )
        expected_size = max(self.config.fallback_size, self.config.min_component_area)

        return not (
            small_target_mode
            and expanded_aspect >= self.config.small_target_expanded_aspect_limit
            and expanded_bbox.width
            >= expected_size * self.config.small_target_expanded_width_multiplier
        )

    def _clean_foreground_mask(self, mask: np.ndarray) -> np.ndarray:
        """Очистить foreground-маску морфологическими операциями."""
        kernel = self._build_morphology_kernel()

        if self.config.expansion_open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self.config.expansion_open_iterations,
            )

        if self.config.expansion_close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.config.expansion_close_iterations,
            )

        return mask

    def _build_morphology_kernel(self) -> np.ndarray:
        """Создать морфологическое ядро для расширения области."""
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self.config.expansion_morphology_kernel,
                self.config.expansion_morphology_kernel,
            ),
        )

    @staticmethod
    def _bbox_from_seed_mask(seed_mask: np.ndarray) -> BoundingBox:
        """Построить bbox непустой области seed-маски."""
        points = np.column_stack(np.where(seed_mask > 0))

        if points.size == 0:
            return BoundingBox(x=0, y=0, width=0, height=0)

        left = int(np.min(points[:, 1]))
        top = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 1])) + 1
        bottom = int(np.max(points[:, 0])) + 1

        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
