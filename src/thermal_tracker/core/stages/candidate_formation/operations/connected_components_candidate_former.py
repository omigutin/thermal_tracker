from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

import cv2

from ....config import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame
from ..result import CandidateFormerResult
from ...motion_localization import MotionLocalizerResult
from ..type import CandidateFormationType
from .base_candidate_former import BaseCandidateFormer


@dataclass(frozen=True, slots=True)
class ConnectedComponentsCandidateFormerConfig:
    """Хранит настройки формирования кандидатов через компоненты связности."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[CandidateFormationType] = CandidateFormationType.CONNECTED_COMPONENTS
    # Минимальная площадь компоненты, из которой можно сформировать кандидата.
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
class ConnectedComponentsCandidateFormer(BaseCandidateFormer):
    """Формирует кандидатов из компонент связности маски движения."""

    config: ConnectedComponentsCandidateFormerConfig

    def apply(
        self,
        frame: ProcessedFrame,
        motion_localizer_result: MotionLocalizerResult,
    ) -> tuple[CandidateFormerResult, ...]:
        """Сформировать кандидатов по компонентам связности маски движения."""
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            motion_localizer_result.mask,
        )

        candidates: list[CandidateFormerResult] = []

        for label in range(1, stats.shape[0]):
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.config.min_area:
                continue

            bbox = BoundingBox(
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                width=int(stats[label, cv2.CC_STAT_WIDTH]),
                height=int(stats[label, cv2.CC_STAT_HEIGHT]),
            ).clamp(frame.bgr.shape)

            candidates.append(
                CandidateFormerResult(
                    bbox=bbox,
                    area=area,
                    confidence=min(1.0, area / max(self.config.min_area, 1)),
                    label="motion_component",
                )
            )

        return tuple(candidates)