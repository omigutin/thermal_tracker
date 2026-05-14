from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self, Optional

import cv2
import numpy as np

from ....config import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame
from ..result import TargetPolarity, TargetSelectorResult
from ..type import TargetSelectorType
from .base_target_selector import BaseTargetSelector


@dataclass(frozen=True, slots=True)
class GrabCutTargetSelectorConfig:
    """Хранит настройки выбора цели через GrabCut по одному клику."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetSelectorType] = TargetSelectorType.GRABCUT

    # Радиус локальной области вокруг клика, внутри которой ищется цель.
    search_radius: int = 48
    # Количество итераций GrabCut.
    iterations: int = 3
    # Радиус уверенного foreground-seed вокруг клика.
    foreground_seed_radius: int = 3
    # Толщина уверенной background-рамки по краям локальной области.
    background_border: int = 3
    # Допуск яркости для пометки probable foreground перед GrabCut.
    intensity_tolerance: int = 18
    # Минимальная площадь компоненты, которую можно принять как цель.
    min_area: int = 16
    # Отступ вокруг итогового bbox.
    padding: int = 2
    # Размер fallback-bbox, если GrabCut не выделил компоненту.
    fallback_size: int = 16
    # Максимальная доля локального patch, которую может занимать результат.
    max_patch_fill: float = 0.75

    # Размер ядра морфологической очистки итоговой маски.
    morphology_kernel: int = 3
    # Количество итераций удаления мелкого шума.
    open_iterations: int = 1
    # Количество итераций заполнения небольших разрывов.
    close_iterations: int = 1

    def __post_init__(self) -> None:
        """Проверить корректность параметров выбора цели."""
        self._validate_positive_int(self.search_radius, "search_radius")
        self._validate_positive_int(self.iterations, "iterations")
        self._validate_positive_int(self.foreground_seed_radius, "foreground_seed_radius")
        self._validate_non_negative_int(self.background_border, "background_border")
        self._validate_non_negative_int(self.intensity_tolerance, "intensity_tolerance")
        self._validate_positive_int(self.min_area, "min_area")
        self._validate_non_negative_int(self.padding, "padding")
        self._validate_positive_int(self.fallback_size, "fallback_size")
        self._validate_ratio(self.max_patch_fill, "max_patch_fill")
        self._validate_odd_positive_kernel(self.morphology_kernel, "morphology_kernel")
        self._validate_non_negative_int(self.open_iterations, "open_iterations")
        self._validate_non_negative_int(self.close_iterations, "close_iterations")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "search_radius")
        reader.pop_int_to(kwargs, "iterations")
        reader.pop_int_to(kwargs, "foreground_seed_radius")
        reader.pop_int_to(kwargs, "background_border")
        reader.pop_int_to(kwargs, "intensity_tolerance")
        reader.pop_int_to(kwargs, "min_area")
        reader.pop_int_to(kwargs, "padding")
        reader.pop_int_to(kwargs, "fallback_size")
        reader.pop_float_to(kwargs, "max_patch_fill")
        reader.pop_int_to(kwargs, "morphology_kernel")
        reader.pop_int_to(kwargs, "open_iterations")
        reader.pop_int_to(kwargs, "close_iterations")
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
    def _validate_ratio(value: float, field_name: str) -> None:
        """Проверить, что значение находится в диапазоне (0, 1]."""
        if not 0 < value <= 1:
            raise ValueError(f"{field_name} must be in range (0, 1].")

    @staticmethod
    def _validate_odd_positive_kernel(value: int, field_name: str) -> None:
        """Проверить, что размер ядра положительный и нечётный."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd.")


@dataclass(slots=True)
class GrabCutTargetSelector(BaseTargetSelector):
    """Выбирает цель по клику через GrabCut-сегментацию локального patch."""

    config: GrabCutTargetSelectorConfig

    def apply(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> TargetSelectorResult:
        """Выделить цель вокруг клика и вернуть bbox найденной области."""
        image = frame.normalized
        frame_shape = frame.bgr.shape
        patch_bbox = self._build_patch_bbox(image_shape=image.shape, point=point, expected_bbox=expected_bbox)
        patch = image[patch_bbox.y:patch_bbox.y2, patch_bbox.x:patch_bbox.x2]

        if patch.size == 0:
            return self._fallback(point=point, frame_shape=frame_shape, patch=None)

        local_x = int(np.clip(point[0] - patch_bbox.x, 0, patch.shape[1] - 1))
        local_y = int(np.clip(point[1] - patch_bbox.y, 0, patch.shape[0] - 1))
        polarity = self._detect_polarity(patch=patch, local_x=local_x, local_y=local_y)

        mask = self._run_grabcut(patch=patch, local_x=local_x, local_y=local_y, polarity=polarity)
        mask = self._clean_mask(mask)

        component = self._extract_clicked_component(mask=mask, local_x=local_x, local_y=local_y)
        if component is None:
            return self._fallback(point=point, frame_shape=frame_shape, patch=patch)

        local_bbox, area = component
        patch_area = patch.shape[0] * patch.shape[1]

        if area < self.config.min_area:
            return self._fallback(point=point, frame_shape=frame_shape, patch=patch)
        if area > int(patch_area * self.config.max_patch_fill):
            return self._fallback(point=point, frame_shape=frame_shape, patch=patch)

        bbox = BoundingBox(
            x=patch_bbox.x + local_bbox.x - self.config.padding,
            y=patch_bbox.y + local_bbox.y - self.config.padding,
            width=local_bbox.width + self.config.padding * 2,
            height=local_bbox.height + self.config.padding * 2,
        ).clamp(frame_shape)

        confidence = min(1.0, area / max(self.config.min_area, 1))

        return TargetSelectorResult(bbox=bbox, confidence=confidence, polarity=polarity)

    def refine(self, frame: ProcessedFrame, bbox: BoundingBox) -> TargetSelectorResult | None:
        """Уточнить bbox цели повторным запуском вокруг центра текущего bbox."""
        result = self.apply(frame=frame, point=(int(bbox.center[0]), int(bbox.center[1])), expected_bbox=bbox)
        if result.bbox.area <= 0:
            return None
        return result

    def _run_grabcut(self, patch: np.ndarray, local_x: int, local_y: int, polarity: TargetPolarity) -> np.ndarray:
        """Запустить GrabCut на локальном patch с автоматически созданными seed."""
        grabcut_image = self._prepare_grabcut_image(patch)
        grabcut_mask = self._build_initial_grabcut_mask(patch=patch, local_x=local_x, local_y=local_y, polarity=polarity)

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        rect = (0, 0, patch.shape[1], patch.shape[0])

        cv2.grabCut(
            grabcut_image,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            self.config.iterations,
            cv2.GC_INIT_WITH_MASK,
        )

        return np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    def _build_initial_grabcut_mask(
        self,
        patch: np.ndarray,
        local_x: int,
        local_y: int,
        polarity: TargetPolarity,
    ) -> np.ndarray:
        """Создать начальную маску GrabCut из клика, рамки фона и яркости."""
        mask = np.full(patch.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

        probable_foreground = self._build_probable_foreground_mask(
            patch=patch,
            local_x=local_x,
            local_y=local_y,
            polarity=polarity,
        )
        mask[probable_foreground > 0] = cv2.GC_PR_FGD

        self._mark_background_border(mask)
        self._mark_foreground_seed(
            mask=mask,
            local_x=local_x,
            local_y=local_y,
        )

        return mask

    def _build_probable_foreground_mask(
        self,
        patch: np.ndarray,
        local_x: int,
        local_y: int,
        polarity: TargetPolarity,
    ) -> np.ndarray:
        """Наметить вероятный foreground по яркости относительно кликнутого пикселя."""
        clicked_value = int(patch[local_y, local_x])
        tolerance = self.config.intensity_tolerance

        if polarity == TargetPolarity.HOT:
            return (patch >= clicked_value - tolerance).astype(np.uint8) * 255

        return (patch <= clicked_value + tolerance).astype(np.uint8) * 255

    def _mark_background_border(self, mask: np.ndarray) -> None:
        """Пометить края локального patch как уверенный background."""
        border = self.config.background_border
        if border <= 0:
            return

        border = min(border, mask.shape[0] // 2, mask.shape[1] // 2)
        if border <= 0:
            return

        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

    def _mark_foreground_seed(self, mask: np.ndarray, local_x: int, local_y: int) -> None:
        """Пометить маленькую область вокруг клика как уверенный foreground."""
        cv2.circle(
            mask,
            center=(local_x, local_y),
            radius=self.config.foreground_seed_radius,
            color=cv2.GC_FGD,
            thickness=-1,
        )

    def _extract_clicked_component(self, mask: np.ndarray, local_x: int, local_y: int) -> tuple[BoundingBox, int] | None:
        """Извлечь компоненту foreground, связанную с кликом или ближайшую к нему."""
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        label = int(labels[local_y, local_x])

        if label == 0:
            label = self._nearest_label(labels=labels, x=local_x, y=local_y)
            if label == 0:
                return None

        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox = BoundingBox(
            x=int(stats[label, cv2.CC_STAT_LEFT]),
            y=int(stats[label, cv2.CC_STAT_TOP]),
            width=int(stats[label, cv2.CC_STAT_WIDTH]),
            height=int(stats[label, cv2.CC_STAT_HEIGHT]),
        )

        return bbox, area

    def _build_patch_bbox(
        self,
        image_shape: tuple[int, int] | tuple[int, int, int],
        point: tuple[int, int],
        expected_bbox: BoundingBox | None,
    ) -> BoundingBox:
        """Построить локальный bbox вокруг клика или ожидаемой области цели."""
        if expected_bbox is not None:
            radius = max(self.config.search_radius, int(max(expected_bbox.width, expected_bbox.height) * 0.75))
        else:
            radius = self.config.search_radius

        x = int(np.clip(point[0], 0, image_shape[1] - 1))
        y = int(np.clip(point[1], 0, image_shape[0] - 1))

        return BoundingBox.from_center(cx=x, cy=y, width=radius * 2 + 1, height=radius * 2 + 1).clamp(image_shape)

    def _fallback(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        patch: Optional[np.ndarray]
    ) -> TargetSelectorResult:
        """Вернуть маленький bbox вокруг клика, если сегментация не сработала."""
        if patch is None or patch.size == 0:
            polarity = TargetPolarity.HOT
        else:
            local_x = int(np.clip(point[0], 0, patch.shape[1] - 1))
            local_y = int(np.clip(point[1], 0, patch.shape[0] - 1))
            polarity = self._detect_polarity(patch=patch, local_x=local_x, local_y=local_y)

        bbox = BoundingBox.from_center(
            point[0],
            point[1],
            self.config.fallback_size,
            self.config.fallback_size,
        ).clamp(frame_shape)

        return TargetSelectorResult(bbox=bbox, confidence=0.05, polarity=polarity)

    def _detect_polarity(self, patch: np.ndarray, local_x: int, local_y: int) -> TargetPolarity:
        """Определить, цель горячее или холоднее локального окружения."""
        clicked_value = float(patch[local_y, local_x])
        median_value = float(np.median(patch))
        if clicked_value >= median_value:
            return TargetPolarity.HOT
        return TargetPolarity.COLD

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Очистить итоговую бинарную маску GrabCut."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )

        if self.config.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations)

        if self.config.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)

        return mask

    @staticmethod
    def _prepare_grabcut_image(patch: np.ndarray) -> np.ndarray:
        """Подготовить 8-битное трёхканальное изображение для GrabCut."""
        if patch.ndim == 2:
            return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

        return patch

    @staticmethod
    def _nearest_label(labels: np.ndarray, x: int, y: int) -> int:
        """Найти ближайшую непустую компоненту, если клик попал в дырку."""
        window = labels[max(0, y - 2): y + 3, max(0, x - 2): x + 3]
        non_zero = window[window > 0]

        if non_zero.size == 0:
            return 0

        values, counts = np.unique(non_zero, return_counts=True)
        return int(values[np.argmax(counts)])
