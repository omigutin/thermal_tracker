from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import cv2
import numpy as np

from .....config import PresetFieldReader
from .....domain.models import ProcessedFrame
from ...type import FramePreprocessorType
from ..base_frame_preprocessor import BaseFramePreprocessor
from ..base_frame_preprocessor_config import BaseFramePreprocessorConfig


@dataclass(frozen=True, slots=True)
class PercentileNormalizeFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки перцентильной нормализации кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.PERCENTILE_NORMALIZE
    # Нижний перцентиль отсечения яркости.
    low_percentile: float = 2.0
    # Верхний перцентиль отсечения яркости.
    high_percentile: float = 98.0

    def __post_init__(self) -> None:
        """Проверить корректность параметров перцентильной нормализации."""
        self._validate_percent(self.low_percentile, "low_percentile")
        self._validate_percent(self.high_percentile, "high_percentile")

        if self.low_percentile >= self.high_percentile:
            raise ValueError("low_percentile must be less than high_percentile.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_float_to(kwargs, "low_percentile")
        reader.pop_float_to(kwargs, "high_percentile")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class PercentileNormalizeFramePreprocessor(BaseFramePreprocessor):
    """Отсекает хвосты яркости gray-канала и записывает результат в normalized."""

    config: PercentileNormalizeFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить перцентильную нормализацию к gray-каналу."""
        gray = frame.gray
        low = float(np.percentile(gray, self.config.low_percentile))
        high = float(np.percentile(gray, self.config.high_percentile))

        if high <= low:
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            clipped = np.clip(gray.astype(np.float32), low, high)
            normalized = ((clipped - low) * (255.0 / (high - low))).astype(np.uint8)

        return replace(frame, normalized=normalized)
