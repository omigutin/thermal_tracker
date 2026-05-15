from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self

import numpy as np

from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader
from ....domain.models import ProcessedFrame
from ...candidate_formation.result import CandidateFormerResult
from ...frame_stabilization.result import FrameStabilizerResult
from ..type import CandidateFilterType
from .base_candidate_filter import BaseCandidateFilter


@dataclass(frozen=True, slots=True)
class ContrastCandidateFilterConfig:
    """Конфигурация фильтра кандидатов по локальному контрасту."""

    # Включает или отключает фильтр
    enabled: bool = True
    # Тип фильтра для связи конфигурации с фабрикой.
    filter_type: ClassVar[CandidateFilterType] = CandidateFilterType.CONTRAST
    # Минимальный контраст объекта относительно локального фона.
    min_contrast: float = 5.0
    # Толщина внешнего кольца вокруг bbox для оценки локального фона.
    border: int = 6
    # Минимальное количество фоновых пикселей для надёжной оценки контраста.
    min_background_pixels: int = 10

    def __post_init__(self) -> None:
        """Проверить корректность параметров фильтра."""
        if self.min_contrast < 0:
            raise ValueError("min_contrast must be greater than or equal to 0.")
        if self.border < 0:
            raise ValueError("border must be greater than or equal to 0.")
        if self.min_background_pixels < 1:
            raise ValueError("min_background_pixels must be greater than or equal to 1.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""

        reader = PresetFieldReader(owner=str(cls.filter_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_float_to(kwargs, "min_contrast")
        reader.pop_int_to(kwargs, "border")
        reader.pop_int_to(kwargs, "min_background_pixels")
        reader.ensure_empty()

        return cls(**kwargs)

@dataclass(frozen=True, slots=True)
class ContrastCandidateFilter(BaseCandidateFilter):
    """
        Фильтр кандидатов по локальному контрасту.
        Модуль содержит атомарный фильтр, который сравнивает яркость объекта с яркостью локального фона вокруг bbox.
        Фильтр полезен для тепловизионных и монохромных кадров, где реальная цель
        обычно должна отличаться от ближайшего окружения по интенсивности.
    """

    config: ContrastCandidateFilterConfig = field(default_factory=ContrastCandidateFilterConfig, )

    def apply(
        self,
        frame: ProcessedFrame,
        objects: list[CandidateFormerResult],
        motion: FrameStabilizerResult,
    ) -> list[CandidateFormerResult]:
        """Удалить объекты с недостаточным контрастом относительно фона."""

        _ = motion

        normalized = frame.normalized.astype(np.float32, copy=False)
        result: list[CandidateFormerResult] = []

        for obj in objects:
            contrast = self._calculate_object_contrast(normalized=normalized, obj=obj, )
            if contrast is None:
                result.append(obj)
                continue
            if contrast >= self.config.min_contrast:
                result.append(obj)

        return result

    def _calculate_object_contrast(self, normalized: np.ndarray, obj: CandidateFormerResult, ) -> float | None:
        """Посчитать контраст объекта относительно локального фона."""

        bbox = obj.bbox.clamp(normalized.shape)
        object_patch = normalized[bbox.y:bbox.y2, bbox.x:bbox.x2]
        if object_patch.size == 0:
            return 0.0

        ring_bbox = bbox.pad(self.config.border, self.config.border, ).clamp(normalized.shape)
        ring_patch = normalized[ring_bbox.y:ring_bbox.y2, ring_bbox.x:ring_bbox.x2]
        background = self._extract_background_ring(ring_patch=ring_patch, bbox=bbox, ring_bbox=ring_bbox, )
        if background.size < self.config.min_background_pixels:
            return None

        return abs(float(np.median(object_patch)) - float(np.median(background)))

    @staticmethod
    def _extract_background_ring(ring_patch: np.ndarray, bbox, ring_bbox, ) -> np.ndarray:
        """Выделить фон вокруг bbox без пикселей самого объекта."""

        # TODO: Маска создаётся на каждого кандидата. Для малого числа кандидатов это нормально,
        #  но при десятках объектов на кадр место может стать горячей точкой.
        ring_mask = np.ones(ring_patch.shape, dtype=bool)

        y1 = bbox.y - ring_bbox.y
        y2 = y1 + bbox.height
        x1 = bbox.x - ring_bbox.x
        x2 = x1 + bbox.width
        ring_mask[y1:y2, x1:x2] = False

        return ring_patch[ring_mask]
