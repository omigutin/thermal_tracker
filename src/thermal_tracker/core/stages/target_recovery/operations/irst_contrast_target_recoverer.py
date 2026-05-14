from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Self

import cv2
import numpy as np

from ....config.preset_field_reader import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame
from ...frame_stabilization import FrameStabilizerResult
from ...target_selection import TargetPolarity
from ..result import TargetRecoveryResult
from ..type import TargetRecovererType
from .base_target_recoverer import BaseTargetRecoverer


@dataclass(frozen=True, slots=True)
class IrstContrastTargetRecovererConfig:
    """Хранит настройки восстановления маленькой тепловой цели по контрасту."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetRecovererType] = TargetRecovererType.IRST_CONTRAST

    # Базовый отступ области поиска вокруг последнего bbox.
    search_padding: int = 30
    # Рост отступа области поиска за каждый потерянный кадр.
    search_padding_growth: int = 4
    # Минимальное значение карты контраста для blob-кандидата.
    contrast_threshold: float = 10.0
    # Минимальная площадь blob-кандидата.
    min_blob_area: int = 1
    # Максимальная площадь blob-кандидата.
    max_blob_area: int = 300
    # Максимальная скорость цели в пикселях за кадр; 0 отключает physical gate.
    max_speed_px_per_frame: float = 0.0
    # Множитель запаса для physical gate.
    physical_gate_lost_frame_multiplier: float = 1.5

    # Размер ядра локального объекта для контрастной карты.
    filter_kernel: int = 3
    # Размер ядра локального фона для контрастной карты.
    background_kernel: int = 11
    # Минимальный размер bbox, если канонический размер ещё неизвестен.
    min_candidate_size: int = 6
    # Размер ядра расширения blob-маски.
    candidate_dilate_kernel: int = 3
    # Количество итераций расширения blob-маски.
    candidate_dilate_iterations: int = 1

    def __post_init__(self) -> None:
        """Проверить корректность параметров IRST recoverer-а."""
        self._validate_non_negative_int(self.search_padding, "search_padding")
        self._validate_non_negative_int(self.search_padding_growth, "search_padding_growth")
        self._validate_non_negative_float(self.contrast_threshold, "contrast_threshold")
        self._validate_positive_int(self.min_blob_area, "min_blob_area")
        self._validate_positive_int(self.max_blob_area, "max_blob_area")
        self._validate_non_negative_float(self.max_speed_px_per_frame, "max_speed_px_per_frame")
        self._validate_positive_float(
            self.physical_gate_lost_frame_multiplier,
            "physical_gate_lost_frame_multiplier",
        )
        self._validate_odd_positive_kernel(self.filter_kernel, "filter_kernel")
        self._validate_odd_positive_kernel(self.background_kernel, "background_kernel")
        self._validate_positive_int(self.min_candidate_size, "min_candidate_size")
        self._validate_odd_positive_kernel(self.candidate_dilate_kernel, "candidate_dilate_kernel")
        self._validate_non_negative_int(
            self.candidate_dilate_iterations,
            "candidate_dilate_iterations",
        )

        if self.min_blob_area > self.max_blob_area:
            raise ValueError("min_blob_area must be less than or equal to max_blob_area.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """
            IRST-recoverer на базе локального контраста.
            Вместо сопоставления шаблонов ищет контрастный blob в расширенной зоне вокруг
            последней известной позиции цели. Запоминает полярность (hot/cold) и
            канонический размер blob — это единственные «шаблонные» данные recoverer-а.
            Используется в связке с IrstSingleTargetTracker через ManualClickTrackingPipeline:
            pipeline вызывает remember() на каждом уверенном TRACKING-кадре и reacquire()
            когда трекер перешёл в SEARCHING и потеря превысила min_lost_frames.
        """
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")

        for field_name in (
            "search_padding",
            "search_padding_growth",
            "min_blob_area",
            "max_blob_area",
            "filter_kernel",
            "background_kernel",
            "min_candidate_size",
            "candidate_dilate_kernel",
            "candidate_dilate_iterations",
        ):
            reader.pop_int_to(kwargs, field_name)

        for field_name in (
            "contrast_threshold",
            "max_speed_px_per_frame",
            "physical_gate_lost_frame_multiplier",
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
    def _validate_odd_positive_kernel(value: int, field_name: str) -> None:
        """Проверить, что размер ядра положительный и нечётный."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd.")


@dataclass(slots=True)
class IrstContrastTargetRecoverer(BaseTargetRecoverer):
    """Восстанавливает маленькую тепловую цель по локальному контрасту."""

    config: IrstContrastTargetRecovererConfig
    _polarity: TargetPolarity = TargetPolarity.HOT
    _canonical_size: tuple[int, int] | None = None

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Запомнить размер и полярность уверенно сопровождаемой цели."""
        self._canonical_size = (bbox.width, bbox.height)
        self._polarity = self._detect_polarity(frame=frame, bbox=bbox)

    def recover(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: FrameStabilizerResult,
        lost_frames: int = 0,
    ) -> TargetRecoveryResult:
        """Найти цель в расширенной зоне вокруг последнего bbox по контрасту."""
        shifted_bbox = self._shift_by_motion(last_bbox, motion)
        search_region = self._build_search_region(
            bbox=shifted_bbox,
            frame_shape=frame.gray.shape,
            lost_frames=lost_frames,
        )
        candidates = self._find_candidates(frame=frame, search_region=search_region)

        if not candidates:
            return TargetRecoveryResult(
                bbox=None,
                search_region=search_region,
                source_name=str(self.config.operation_type),
                message="No contrast candidates were found.",
            )

        candidates = self._filter_by_physical_gate(
            candidates=candidates,
            reference_bbox=shifted_bbox,
            lost_frames=lost_frames,
        )

        if not candidates:
            return TargetRecoveryResult(
                bbox=None,
                search_region=search_region,
                source_name=str(self.config.operation_type),
                message="Contrast candidates were rejected by physical gate.",
            )

        best_bbox, best_score = self._select_best_candidate(
            candidates=candidates,
            reference_bbox=shifted_bbox,
        )

        return TargetRecoveryResult(
            bbox=best_bbox,
            score=best_score,
            search_region=search_region,
            source_name=str(self.config.operation_type),
            message="Target recovered by IRST contrast.",
        )

    def reset(self) -> None:
        """Сбросить память о цели."""
        self._canonical_size = None
        self._polarity = TargetPolarity.HOT

    def _find_candidates(
        self,
        frame: ProcessedFrame,
        search_region: BoundingBox,
    ) -> list[tuple[BoundingBox, float]]:
        """Найти контрастные blob-кандидаты внутри области поиска."""
        gray = frame.gray
        contrast_map = self._compute_contrast_map(gray)

        region = search_region.clamp(gray.shape)
        search_mask = np.zeros(contrast_map.shape, dtype=np.float32)
        search_mask[region.y:region.y2, region.x:region.x2] = 1.0
        masked_contrast = contrast_map * search_mask

        binary = (masked_contrast >= self.config.contrast_threshold).astype(np.uint8) * 255

        if self.config.candidate_dilate_iterations > 0:
            kernel = np.ones(
                (
                    self.config.candidate_dilate_kernel,
                    self.config.candidate_dilate_kernel,
                ),
                dtype=np.uint8,
            )
            binary = cv2.dilate(
                binary,
                kernel,
                iterations=self.config.candidate_dilate_iterations,
            )

        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        candidates: list[tuple[BoundingBox, float]] = []

        for label in range(1, label_count):
            area = int(stats[label, cv2.CC_STAT_AREA])

            if not self.config.min_blob_area <= area <= self.config.max_blob_area:
                continue

            center_x = float(centroids[label][0])
            center_y = float(centroids[label][1])
            bbox_width, bbox_height = self._resolve_candidate_size(stats=stats, label=label)

            bbox = BoundingBox.from_center(
                center_x,
                center_y,
                bbox_width,
                bbox_height,
            ).clamp(gray.shape)

            blob_pixels = masked_contrast[labels == label]
            score = float(np.mean(blob_pixels)) / 255.0 if blob_pixels.size > 0 else 0.0

            candidates.append((bbox, score))

        candidates.sort(key=lambda candidate: -candidate[1])
        return candidates

    def _compute_contrast_map(self, gray: np.ndarray) -> np.ndarray:
        """Вычислить карту локального контраста с учётом полярности цели."""
        gray_float = gray.astype(np.float32)
        inner_kernel = np.ones(
            (self.config.filter_kernel, self.config.filter_kernel),
            dtype=np.uint8,
        )
        local_max = cv2.dilate(gray, inner_kernel).astype(np.float32)
        local_min = cv2.erode(gray, inner_kernel).astype(np.float32)
        background = cv2.blur(
            gray_float,
            (self.config.background_kernel, self.config.background_kernel),
        )

        if self._polarity == TargetPolarity.COLD:
            contrast = background - local_min
        else:
            contrast = local_max - background

        return np.clip(contrast, 0.0, 255.0)

    def _build_search_region(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        lost_frames: int,
    ) -> BoundingBox:
        """Построить область поиска с ростом по длительности потери."""
        effective_padding = (
            self.config.search_padding
            + max(0, int(lost_frames)) * self.config.search_padding_growth
        )

        return bbox.pad(effective_padding, effective_padding).clamp(frame_shape)

    def _filter_by_physical_gate(
        self,
        candidates: list[tuple[BoundingBox, float]],
        reference_bbox: BoundingBox,
        lost_frames: int,
    ) -> list[tuple[BoundingBox, float]]:
        """Отфильтровать кандидатов по физически допустимому смещению."""
        if self.config.max_speed_px_per_frame <= 0.0:
            return candidates

        reference_x, reference_y = reference_bbox.center
        max_distance = (
            self.config.max_speed_px_per_frame
            * max(lost_frames, 1)
            * self.config.physical_gate_lost_frame_multiplier
        )

        return [
            (bbox, score)
            for bbox, score in candidates
            if math.hypot(
                bbox.center[0] - reference_x,
                bbox.center[1] - reference_y,
            )
            <= max_distance
        ]

    def _select_best_candidate(
        self,
        candidates: list[tuple[BoundingBox, float]],
        reference_bbox: BoundingBox,
    ) -> tuple[BoundingBox, float]:
        """Выбрать кандидата по score с мягким штрафом за расстояние."""
        reference_x, reference_y = reference_bbox.center
        reference_distance = max(self.config.search_padding / 4.0, 1.0)

        return max(
            candidates,
            key=lambda candidate: candidate[1]
            / (
                1.0
                + math.hypot(
                    candidate[0].center[0] - reference_x,
                    candidate[0].center[1] - reference_y,
                )
                / reference_distance
            ),
        )

    def _resolve_candidate_size(
        self,
        stats: np.ndarray,
        label: int,
    ) -> tuple[int, int]:
        """Вернуть размер bbox-кандидата."""
        if self._canonical_size is not None:
            return self._canonical_size

        return (
            max(int(stats[label, cv2.CC_STAT_WIDTH]), self.config.min_candidate_size),
            max(int(stats[label, cv2.CC_STAT_HEIGHT]), self.config.min_candidate_size),
        )

    @staticmethod
    def _shift_by_motion(
        bbox: BoundingBox,
        motion: FrameStabilizerResult,
    ) -> BoundingBox:
        """Сместить bbox на глобальное движение камеры, если оно валидно."""
        if not motion.valid:
            return bbox

        center_x, center_y = bbox.center

        return BoundingBox.from_center(
            center_x + float(motion.dx),
            center_y + float(motion.dy),
            bbox.width,
            bbox.height,
        )

    @staticmethod
    def _detect_polarity(frame: ProcessedFrame, bbox: BoundingBox) -> TargetPolarity:
        """Определить, цель горячее или холоднее локального фона."""
        gray = frame.gray
        clamped = bbox.clamp(gray.shape)
        object_patch = gray[clamped.y:clamped.y2, clamped.x:clamped.x2]

        if object_patch.size == 0:
            return TargetPolarity.HOT

        margin = max(4, min(clamped.width, clamped.height))
        outer = clamped.pad(margin, margin).clamp(gray.shape)
        outer_patch = gray[outer.y:outer.y2, outer.x:outer.x2]

        if outer_patch.size == 0:
            return TargetPolarity.HOT

        ring_mask = np.ones(outer_patch.shape, dtype=bool)
        inner_y1 = clamped.y - outer.y
        inner_x1 = clamped.x - outer.x
        ring_mask[
            inner_y1:inner_y1 + clamped.height,
            inner_x1:inner_x1 + clamped.width,
        ] = False

        background_values = outer_patch[ring_mask]

        if background_values.size == 0:
            return TargetPolarity.HOT

        if float(np.mean(object_patch)) >= float(np.mean(background_values)):
            return TargetPolarity.HOT

        return TargetPolarity.COLD
