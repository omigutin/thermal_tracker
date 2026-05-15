from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import cv2

from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader
from .....domain.models import ProcessedFrame
from ...type import FramePreprocessorType
from ..base_frame_preprocessor import BaseFramePreprocessor
from ..base_frame_preprocessor_config import BaseFramePreprocessorConfig


@dataclass(frozen=True, slots=True)
class MedianBlurFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки медианного сглаживания кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.MEDIAN_BLUR
    # Размер ядра медианного фильтра; 1 отключает фактическое сглаживание.
    kernel: int = 3

    def __post_init__(self) -> None:
        """Проверить корректность параметров медианного сглаживания."""
        self._validate_odd_positive_kernel(self.kernel, "kernel")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "kernel")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class MedianBlurFramePreprocessor(BaseFramePreprocessor):
    """Сглаживает gray-канал медианным фильтром."""

    config: MedianBlurFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить медианное сглаживание к gray-каналу."""
        if self.config.kernel <= 1:
            return frame

        gray = cv2.medianBlur(frame.gray, self.config.kernel)
        return replace(frame, gray=gray)
