from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import cv2
import numpy as np

from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader
from .....domain.models import FrameQuality, ProcessedFrame
from ...type import FramePreprocessorType
from ..base_frame_preprocessor import BaseFramePreprocessor
from ..base_frame_preprocessor_config import BaseFramePreprocessorConfig


@dataclass(frozen=True, slots=True)
class SharpnessMetricFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки расчёта метрики резкости кадра."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.SHARPNESS_METRIC
    # Левая граница ROI по ширине в долях кадра.
    crop_left: float = 0.08
    # Правая граница ROI по ширине в долях кадра.
    crop_right: float = 0.92
    # Верхняя граница ROI по высоте в долях кадра.
    crop_top: float = 0.18
    # Нижняя граница ROI по высоте в долях кадра.
    crop_bottom: float = 0.82
    # Перцентиль абсолютного Лапласиана для итоговой метрики.
    percentile: float = 90.0
    # Размер ядра Лапласиана.
    laplacian_kernel: int = 3

    def __post_init__(self) -> None:
        """Проверить корректность параметров метрики резкости."""
        self._validate_ratio(self.crop_left, "crop_left")
        self._validate_ratio(self.crop_right, "crop_right")
        self._validate_ratio(self.crop_top, "crop_top")
        self._validate_ratio(self.crop_bottom, "crop_bottom")
        self._validate_percent(self.percentile, "percentile")
        self._validate_sobel_like_kernel(self.laplacian_kernel, "laplacian_kernel")

        if self.crop_left >= self.crop_right:
            raise ValueError("crop_left must be less than crop_right.")
        if self.crop_top >= self.crop_bottom:
            raise ValueError("crop_top must be less than crop_bottom.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_float_to(kwargs, "crop_left")
        reader.pop_float_to(kwargs, "crop_right")
        reader.pop_float_to(kwargs, "crop_top")
        reader.pop_float_to(kwargs, "crop_bottom")
        reader.pop_float_to(kwargs, "percentile")
        reader.pop_int_to(kwargs, "laplacian_kernel")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class SharpnessMetricFramePreprocessor(BaseFramePreprocessor):
    """Считает резкость кадра и записывает значение в ProcessedFrame.quality."""

    config: SharpnessMetricFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Рассчитать sharpness-метрику по центральной ROI кадра."""
        gray = frame.gray
        frame_height, frame_width = gray.shape[:2]
        x1 = int(round(frame_width * self.config.crop_left))
        x2 = int(round(frame_width * self.config.crop_right))
        y1 = int(round(frame_height * self.config.crop_top))
        y2 = int(round(frame_height * self.config.crop_bottom))
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            roi = gray

        laplacian = cv2.Laplacian(
            roi,
            cv2.CV_32F,
            ksize=self.config.laplacian_kernel,
        )
        sharpness = float(np.percentile(np.abs(laplacian), self.config.percentile))

        if frame.quality is None:
            quality = FrameQuality(sharpness=sharpness, blurred=False)
        else:
            quality = replace(frame.quality, sharpness=sharpness)

        return replace(frame, quality=quality)
