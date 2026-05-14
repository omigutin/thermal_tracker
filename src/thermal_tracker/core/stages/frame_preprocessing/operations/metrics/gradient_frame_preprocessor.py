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
class GradientFramePreprocessorConfig(BaseFramePreprocessorConfig):
    """Хранит настройки расчёта карты градиентов."""

    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FramePreprocessorType] = FramePreprocessorType.GRADIENT
    # Размер ядра размытия перед Sobel; 1 отключает размытие.
    blur_kernel: int = 3
    # Размер ядра Sobel для расчёта градиента.
    sobel_kernel: int = 3

    def __post_init__(self) -> None:
        """Проверить корректность параметров градиента."""
        self._validate_odd_positive_kernel(self.blur_kernel, "blur_kernel")
        self._validate_sobel_like_kernel(self.sobel_kernel, "sobel_kernel")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "blur_kernel")
        reader.pop_int_to(kwargs, "sobel_kernel")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class GradientFramePreprocessor(BaseFramePreprocessor):
    """Считает карту градиентов из normalized-канала."""

    config: GradientFramePreprocessorConfig

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Рассчитать gradient-канал по normalized-каналу."""
        work = frame.normalized

        if self.config.blur_kernel > 1:
            work = cv2.GaussianBlur(
                work,
                (self.config.blur_kernel, self.config.blur_kernel),
                0,
            )

        grad_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=self.config.sobel_kernel)
        grad_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=self.config.sobel_kernel)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return replace(frame, gradient=gradient)
