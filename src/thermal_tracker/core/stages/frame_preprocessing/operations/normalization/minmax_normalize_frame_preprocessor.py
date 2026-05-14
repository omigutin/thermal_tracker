from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import cv2

from .....config.preset_field_reader import PresetFieldReader
from .....domain.models import ProcessedFrame
from ...type import FramePreprocessorType
from ..base_frame_preprocessor import BaseFramePreprocessor
from ..base_frame_preprocessor_config import BaseFramePreprocessorConfig


@dataclass(frozen=True, slots=True)
class MinMaxNormalizeFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки min-max нормализации кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.MINMAX_NORMALIZE

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class MinMaxNormalizeFramePreprocessor(BaseFramePreprocessor):
    """Линейно растягивает gray-канал в normalized-канал диапазона 0..255."""

    config: MinMaxNormalizeFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить min-max нормализацию к gray-каналу."""
        normalized = cv2.normalize(frame.gray, None, 0, 255, cv2.NORM_MINMAX)
        return replace(frame, normalized=normalized)
