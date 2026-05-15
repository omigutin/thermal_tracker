from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self, Optional

import cv2
import numpy as np

from ..result import MotionLocalizerResult
from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader
from ....domain.models import ProcessedFrame
from ..type import MotionLocalizationType
from .base_motion_localizer import BaseMotionLocalizer, BaseMotionLocalizerConfig


@dataclass(frozen=True, slots=True)
class FrameDifferenceMotionLocalizerConfig(BaseMotionLocalizerConfig):
    """Хранит настройки локализации движения через разность соседних кадров."""

    # Включает или отключает операцию
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой
    operation_type: ClassVar[MotionLocalizationType] = MotionLocalizationType.FRAME_DIFFERENCE
    # Минимальная разница яркости пикселей, которая считается движением
    threshold: int = 22
    # Размер ядра размытия перед бинаризацией
    blur_kernel: int = 5
    # Размер ядра морфологической очистки маски
    morphology_kernel: int = 5
    # Количество итераций удаления мелкого шума
    open_iterations: int = 1
    # Количество итераций заполнения небольших разрывов
    close_iterations: int = 2

    def __post_init__(self) -> None:
        """Проверить корректность параметров локализации движения."""
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be in range [0, 255].")
        self.validate_odd_positive_kernel(self.blur_kernel, "blur_kernel")
        self.validate_odd_positive_kernel(self.morphology_kernel, "morphology_kernel")
        if self.open_iterations < 0:
            raise ValueError("open_iterations must be greater than or equal to 0.")
        if self.close_iterations < 0:
            raise ValueError("close_iterations must be greater than or equal to 0.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "threshold")
        reader.pop_int_to(kwargs, "blur_kernel")
        reader.pop_int_to(kwargs, "morphology_kernel")
        reader.pop_int_to(kwargs, "open_iterations")
        reader.pop_int_to(kwargs, "close_iterations")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class FrameDifferenceMotionLocalizer(BaseMotionLocalizer):
    """Локализует движение по абсолютной разности соседних нормализованных кадров."""

    config: FrameDifferenceMotionLocalizerConfig
    _previous: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def apply(self, frame: ProcessedFrame) -> MotionLocalizerResult:
        """Построить маску движения для текущего кадра."""
        current = frame.normalized

        # На первом кадре маска пустая
        if self._previous is None:
            self._previous = current.copy()
            return MotionLocalizerResult(
                mask=np.zeros_like(current, dtype=np.uint8),
                confidence_map=np.zeros_like(current, dtype=np.uint8),
                motion_score=0.0,
            )

        difference = cv2.absdiff(current, self._previous)
        self._previous = current.copy()

        if self.config.blur_kernel > 1:
            difference = cv2.GaussianBlur(difference, (self.config.blur_kernel, self.config.blur_kernel),0)

        _, mask = cv2.threshold(difference, self.config.threshold,255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )

        if self.config.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations)

        if self.config.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)

        motion_score = float(np.count_nonzero(mask)) / max(mask.size, 1)

        return MotionLocalizerResult(mask=mask, confidence_map=difference, motion_score=motion_score)
