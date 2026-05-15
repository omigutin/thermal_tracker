from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Self

import cv2

from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader
from .....domain.models import ProcessedFrame
from ...type import FramePreprocessorType
from ..base_frame_preprocessor import BaseFramePreprocessor
from ..base_frame_preprocessor_config import BaseFramePreprocessorConfig


@dataclass(frozen=True, slots=True)
class ResizeFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки масштабирования кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.RESIZE
    # Целевая ширина кадра; None означает отсутствие масштабирования.
    target_width: int | None = 960

    def __post_init__(self) -> None:
        """Проверить корректность параметров масштабирования."""
        if self.target_width is not None:
            self._validate_positive_int(self.target_width, "target_width")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "target_width")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class ResizeFramePreprocessor(BaseFramePreprocessor):
    """Уменьшает все каналы кадра до заданной ширины с сохранением пропорций."""

    config: ResizeFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить масштабирование ко всем каналам ProcessedFrame."""
        target_width = self.config.target_width

        if target_width is None or frame.bgr.shape[1] <= target_width:
            return frame

        scale = target_width / frame.bgr.shape[1]
        target_height = int(round(frame.bgr.shape[0] * scale))
        size = (target_width, target_height)

        return ProcessedFrame(
            bgr=cv2.resize(frame.bgr, size, interpolation=cv2.INTER_AREA),
            gray=cv2.resize(frame.gray, size, interpolation=cv2.INTER_AREA),
            normalized=cv2.resize(frame.normalized, size, interpolation=cv2.INTER_AREA),
            gradient=cv2.resize(frame.gradient, size, interpolation=cv2.INTER_AREA),
            quality=frame.quality,
        )
