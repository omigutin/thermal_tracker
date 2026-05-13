from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self

from ....config import PresetFieldReader
from ....domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from ..type import CandidateFilterType
from .base_candidate_filter import BaseCandidateFilter


@dataclass(frozen=True, slots=True)
class BorderTouchCandidateFilterConfig:
    """Конфигурация фильтра кандидатов по близости bbox к границам кадра."""

    # Включает или отключает фильтр
    enabled: bool = True
    # Тип фильтра для связи конфигурации с фабрикой.
    filter_type: ClassVar[CandidateFilterType] = CandidateFilterType.BORDER_TOUCH
    # Минимальный отступ bbox от границы кадра в пикселях.
    border_margin: int = 2

    def __post_init__(self) -> None:
        """Проверить корректность параметров фильтра."""
        if self.border_margin < 0:
            raise ValueError("border_margin must be greater than or equal to 0.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.filter_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "border_margin")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class BorderTouchCandidateFilter(BaseCandidateFilter):
    """
        Фильтр кандидатов по касанию границ кадра.
        Модуль содержит атомарный фильтр, который отбрасывает объекты-кандидаты,
        если их bbox находится слишком близко к границе кадра.
        Такие объекты часто являются неполными, обрезанными или нестабильными ложными срабатываниями.
    """

    config: BorderTouchCandidateFilterConfig = field(default_factory=BorderTouchCandidateFilterConfig, )

    def filter(
        self,
        frame: ProcessedFrame,
        objects: list[DetectedObject],
        motion: GlobalMotion,
    ) -> list[DetectedObject]:
        """Удалить объекты, расположенные слишком близко к границе кадра."""

        _ = motion

        frame_h, frame_w = frame.bgr.shape[:2]
        result: list[DetectedObject] = []

        for obj in objects:
            if obj.bbox.x <= self.config.border_margin:
                continue
            if obj.bbox.y <= self.config.border_margin:
                continue
            if obj.bbox.x2 >= frame_w - self.config.border_margin:
                continue
            if obj.bbox.y2 >= frame_h - self.config.border_margin:
                continue
            result.append(obj)

        return result
