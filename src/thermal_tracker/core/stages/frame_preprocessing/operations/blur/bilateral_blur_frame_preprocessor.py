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
class BilateralBlurFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки bilateral-сглаживания кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.BILATERAL_BLUR
    # Диаметр окрестности фильтрации.
    diameter: int = 7
    # Допуск различия яркости пикселей внутри фильтра.
    sigma_color: float = 40.0
    # Допуск пространственного расстояния между пикселями.
    sigma_space: float = 40.0

    def __post_init__(self) -> None:
        """Проверить корректность параметров bilateral-сглаживания."""
        self._validate_positive_int(self.diameter, "diameter")
        self._validate_positive_float(self.sigma_color, "sigma_color")
        self._validate_positive_float(self.sigma_space, "sigma_space")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "diameter")
        reader.pop_float_to(kwargs, "sigma_color")
        reader.pop_float_to(kwargs, "sigma_space")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class BilateralBlurFramePreprocessor(BaseFramePreprocessor):
    """Сглаживает gray-канал bilateral-фильтром с сохранением границ."""

    config: BilateralBlurFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Применить bilateral-сглаживание к gray-каналу."""
        gray = cv2.bilateralFilter(
            frame.gray,
            self.config.diameter,
            self.config.sigma_color,
            self.config.sigma_space,
        )
        return replace(frame, gray=gray)
