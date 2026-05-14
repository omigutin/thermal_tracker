from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import cv2

from .....config import PresetFieldReader
from .....domain.models import ProcessedFrame
from ...type import FramePreprocessorType
from ..base_frame_preprocessor import BaseFramePreprocessor
from ..base_frame_preprocessor_config import BaseFramePreprocessorConfig


@dataclass(frozen=True, slots=True)
class GaussianBlurFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки гауссова сглаживания кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.GAUSSIAN_BLUR
    # Размер ядра гауссова фильтра; 1 отключает фактическое сглаживание.
    kernel: int = 3

    def __post_init__(self) -> None:
        """Проверить корректность параметров гауссова сглаживания."""
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
class GaussianBlurFramePreprocessor(BaseFramePreprocessor):
    """Сглаживает gray-канал гауссовым фильтром."""

    config: GaussianBlurFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить гауссово сглаживание к gray-каналу."""
        if self.config.kernel <= 1:
            return frame

        gray = cv2.GaussianBlur(
            frame.gray,
            (self.config.kernel, self.config.kernel),
            0,
        )
        return replace(frame, gray=gray)
