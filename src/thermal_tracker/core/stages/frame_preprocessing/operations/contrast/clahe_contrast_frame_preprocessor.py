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
class ClaheContrastFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки усиления локального контраста через CLAHE."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.CLAHE_CONTRAST
    # Ограничение усиления локального контраста.
    clip_limit: float = 2.0
    # Размер сетки CLAHE по одной стороне.
    tile_grid_size: int = 8

    def __post_init__(self) -> None:
        """Проверить корректность параметров CLAHE."""
        self._validate_positive_float(self.clip_limit, "clip_limit")
        self._validate_positive_int(self.tile_grid_size, "tile_grid_size")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_float_to(kwargs, "clip_limit")
        reader.pop_int_to(kwargs, "tile_grid_size")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class ClaheContrastFramePreprocessor(BaseFramePreprocessor):
    """Усиливает локальный контраст normalized-канала через CLAHE."""

    config: ClaheContrastFramePreprocessorConfig

    def __post_init__(self) -> None:
        """Создать CLAHE-фильтр один раз при инициализации операции."""
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=(self.config.tile_grid_size, self.config.tile_grid_size),
        )

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить CLAHE к normalized-каналу."""
        normalized = self._clahe.apply(frame.normalized)
        return replace(frame, normalized=normalized)
