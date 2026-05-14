from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self

import cv2
import numpy as np

from ....config import PresetFieldReader
from ....domain.models import ProcessedFrame
from ..result import MotionLocalizerResult
from ..type import MotionLocalizationType
from .base_motion_localizer import BaseMotionLocalizer, BaseMotionLocalizerConfig


@dataclass(frozen=True, slots=True)
class Mog2MotionLocalizerConfig(BaseMotionLocalizerConfig):
    """Хранит настройки локализации движения через MOG2 background subtraction."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[MotionLocalizationType] = MotionLocalizationType.MOG2
    # Количество кадров, используемых для построения фоновой модели.
    history: int = 300
    # Порог чувствительности MOG2 к изменениям пикселей.
    var_threshold: float = 20.0
    # Включает обработку теней внутри OpenCV-алгоритма.
    detect_shadows: bool = False
    # Порог бинаризации сырой маски движения.
    threshold: int = 200
    # Размер ядра морфологической очистки маски.
    morphology_kernel: int = 5
    # Количество итераций удаления мелкого шума.
    open_iterations: int = 1
    # Количество итераций заполнения небольших разрывов.
    close_iterations: int = 2

    def __post_init__(self) -> None:
        """Проверить корректность параметров локализации движения."""
        if self.history <= 0:
            raise ValueError("history must be greater than 0.")
        if self.var_threshold <= 0:
            raise ValueError("var_threshold must be greater than 0.")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be in range [0, 255].")

        self.validate_odd_positive_kernel(value=self.morphology_kernel, field_name="morphology_kernel")

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
        reader.pop_int_to(kwargs, "history")
        reader.pop_float_to(kwargs, "var_threshold")
        reader.pop_bool_to(kwargs, "detect_shadows")
        reader.pop_int_to(kwargs, "threshold")
        reader.pop_int_to(kwargs, "morphology_kernel")
        reader.pop_int_to(kwargs, "open_iterations")
        reader.pop_int_to(kwargs, "close_iterations")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class Mog2MotionLocalizer(BaseMotionLocalizer):
    """Локализует движение через MOG2 background subtraction."""

    config: Mog2MotionLocalizerConfig
    _subtractor: cv2.BackgroundSubtractor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Создать OpenCV-модель вычитания фона."""
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.history,
            varThreshold=self.config.var_threshold,
            detectShadows=self.config.detect_shadows,
        )

    def apply(self, frame: ProcessedFrame) -> MotionLocalizerResult:
        """Построить маску движения для текущего кадра."""
        raw_mask = self._subtractor.apply(frame.normalized)

        _, mask = cv2.threshold(raw_mask, self.config.threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )

        if self.config.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.open_iterations)

        if self.config.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.close_iterations)

        motion_score = float(np.count_nonzero(mask)) / max(mask.size, 1)

        return MotionLocalizerResult(mask=mask, confidence_map=raw_mask, motion_score=motion_score)
