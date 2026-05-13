from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ....config import PresetFieldReader
from ....domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from ..type import CandidateFilterType
from .base_candidate_filter import BaseCandidateFilter


@dataclass(frozen=True, slots=True)
class AreaAspectCandidateFilterConfig:
    """Конфигурация фильтра кандидатов по площади и пропорциям bbox."""

    # Включает или отключает фильтр
    enabled: bool = True
    # Тип фильтра для связи конфигурации с фабрикой.
    filter_type: ClassVar[CandidateFilterType] = CandidateFilterType.AREA_ASPECT
    # Минимальная допустимая площадь кандидата в пикселях.
    min_area: int = 24
    # Максимальная доля площади кадра, которую может занимать кандидат.
    max_frame_fraction: float = 0.25
    # Максимальное допустимое соотношение длинной и короткой стороны bbox.
    max_aspect_ratio: float = 6.0

    def __post_init__(self) -> None:
        """Проверить корректность параметров фильтра."""
        if self.min_area < 0:
            raise ValueError("min_area must be greater than or equal to 0.")
        if not 0 < self.max_frame_fraction <= 1:
            raise ValueError("max_frame_fraction must be in range (0, 1].")
        if self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be greater than or equal to 1.")

    @classmethod
    def from_mapping(cls, values: dict[str, object], ) -> AreaAspectCandidateFilterConfig:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.filter_type), values=values, )
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "min_area")
        reader.pop_float_to(kwargs, "max_frame_fraction")
        reader.pop_float_to(kwargs, "max_aspect_ratio")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class AreaAspectCandidateFilter(BaseCandidateFilter):
    """
        Фильтр кандидатов по площади и пропорциям bbox.
        Модуль содержит атомарный фильтр, который отбрасывает объекты-кандидаты по простым геометрическим признакам:
        - слишком маленькая площадь;
        - слишком большая площадь относительно кадра;
        - слишком вытянутая форма bbox.
        Фильтр полезен как ранний дешёвый этап очистки ложных срабатываний.
    """

    config: AreaAspectCandidateFilterConfig = field(default_factory=AreaAspectCandidateFilterConfig, )

    def filter(self,
        frame: ProcessedFrame,
        objects: list[DetectedObject],
        motion: GlobalMotion,
    ) -> list[DetectedObject]:
        """Удалить объекты с недопустимой площадью или пропорциями bbox."""

        _ = motion

        frame_area = frame.bgr.shape[0] * frame.bgr.shape[1]
        result: list[DetectedObject] = []

        for obj in objects:
            if obj.area < self.config.min_area:
                continue
            if obj.area > frame_area * self.config.max_frame_fraction:
                continue
            aspect_ratio = self._calculate_aspect_ratio(width=obj.bbox.width, height=obj.bbox.height, )
            if aspect_ratio > self.config.max_aspect_ratio:
                continue
            result.append(obj)

        return result

    @staticmethod
    def _calculate_aspect_ratio(width: int, height: int) -> float:
        """Посчитать отношение длинной стороны bbox к короткой."""
        long_side = max(width, height)
        short_side = max(min(width, height), 1)
        return long_side / short_side
