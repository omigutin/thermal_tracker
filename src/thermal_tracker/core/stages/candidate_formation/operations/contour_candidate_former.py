from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

import cv2

from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame
from ...motion_localization import MotionLocalizerResult
from ..result import CandidateFormerResult
from ..type import CandidateFormationType
from .base_candidate_former import BaseCandidateFormer


@dataclass(frozen=True, slots=True)
class ContourCandidateFormerConfig:
    """Хранит настройки формирования кандидатов через внешние контуры."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[CandidateFormationType] = CandidateFormationType.CONTOUR
    # Минимальная площадь контура, из которого можно сформировать кандидата.
    min_area: int = 24

    def __post_init__(self) -> None:
        """Проверить корректность параметров формирования кандидатов."""
        if self.min_area < 0:
            raise ValueError("min_area must be greater than or equal to 0.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "min_area")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class ContourCandidateFormer(BaseCandidateFormer):
    """Формирует кандидатов по внешним контурам маски движения."""

    config: ContourCandidateFormerConfig

    def apply(
        self,
        frame: ProcessedFrame,
        motion_localizer_result: MotionLocalizerResult,
    ) -> tuple[CandidateFormerResult, ...]:
        """Сформировать кандидатов по внешним контурам маски движения."""
        contours, _ = cv2.findContours(motion_localizer_result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[CandidateFormerResult] = []

        for contour in contours:
            area = int(cv2.contourArea(contour))

            if area < self.config.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            bbox = BoundingBox(x=x, y=y, width=width, height=height).clamp(frame.bgr.shape)

            candidates.append(
                CandidateFormerResult(
                    bbox=bbox,
                    area=area,
                    confidence=min(1.0, area / max(self.config.min_area, 1)),
                    label="motion_contour",
                )
            )

        return tuple(candidates)
