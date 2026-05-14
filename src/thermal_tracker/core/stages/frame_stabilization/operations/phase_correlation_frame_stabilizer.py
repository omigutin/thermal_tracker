from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self, Optional

import cv2
import numpy as np

from ....config.preset_field_reader import PresetFieldReader
from ....domain.models import ProcessedFrame
from ..result import FrameStabilizerResult
from ..type import FrameStabilizerType
from .base_frame_stabilizer import BaseFrameStabilizer


@dataclass(frozen=True, slots=True)
class PhaseCorrelationFrameStabilizerConfig:
    """Хранит настройки стабилизации кадра через фазовую корреляцию."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FrameStabilizerType] = FrameStabilizerType.PHASE_CORRELATION
    # Масштаб кадра перед оценкой смещения.
    downscale: float = 0.5
    # Размер ядра размытия перед фазовой корреляцией.
    blur_kernel: int = 9
    # Минимальная уверенность phase correlation для признания результата корректным.
    min_response: float = 0.03
    # Максимально допустимый сдвиг относительно размера кадра.
    max_shift_ratio: float = 0.35

    def __post_init__(self) -> None:
        """Проверить корректность параметров стабилизации кадра."""
        if not 0 < self.downscale <= 1:
            raise ValueError("downscale must be in range (0, 1].")
        if self.blur_kernel < 1:
            raise ValueError("blur_kernel must be greater than or equal to 1.")
        if self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd.")
        if self.min_response < 0:
            raise ValueError("min_response must be greater than or equal to 0.")
        if not 0 < self.max_shift_ratio <= 1:
            raise ValueError("max_shift_ratio must be in range (0, 1].")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_float_to(kwargs, "downscale")
        reader.pop_int_to(kwargs, "blur_kernel")
        reader.pop_float_to(kwargs, "min_response")
        reader.pop_float_to(kwargs, "max_shift_ratio")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class PhaseCorrelationFrameStabilizer(BaseFrameStabilizer):
    """Оценивает сдвиг кадра через фазовую корреляцию."""

    config: PhaseCorrelationFrameStabilizerConfig
    _previous: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _window: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def apply(self, frame: ProcessedFrame) -> FrameStabilizerResult:
        """Вернуть оценку смещения текущего кадра относительно предыдущего."""
        if not self.config.enabled:
            return FrameStabilizerResult()

        current = self._prepare_frame(frame)

        if self._window is None or self._window.shape != current.shape:
            self._window = cv2.createHanningWindow((current.shape[1], current.shape[0]), cv2.CV_32F)

        if self._previous is None:
            self._previous = current
            return FrameStabilizerResult()

        (dx, dy), response = cv2.phaseCorrelate(self._previous, current, self._window)
        self._previous = current

        scaled_dx = float(dx / self.config.downscale)
        scaled_dy = float(dy / self.config.downscale)
        response = float(response)

        if not self._is_valid_result(frame=frame, dx=scaled_dx, dy=scaled_dy, response=response):
            return FrameStabilizerResult(response=response, valid=False)

        return FrameStabilizerResult(dx=scaled_dx, dy=scaled_dy, response=response, valid=True)

    def _prepare_frame(self, frame: ProcessedFrame) -> np.ndarray:
        """Подготовить кадр для фазовой корреляции."""
        current = frame.normalized.astype(np.float32) / 255.0

        if self.config.downscale != 1.0:
            current = cv2.resize(
                current,
                None,
                fx=self.config.downscale,
                fy=self.config.downscale,
                interpolation=cv2.INTER_AREA,
            )

        if self.config.blur_kernel > 1:
            current = cv2.GaussianBlur(current, (self.config.blur_kernel, self.config.blur_kernel),0)

        return current

    def _is_valid_result(self, frame: ProcessedFrame, dx: float, dy: float, response: float) -> bool:
        """Проверить, можно ли использовать рассчитанное смещение."""
        max_shift = max(frame.bgr.shape[:2]) * self.config.max_shift_ratio
        return response >= self.config.min_response and abs(dx) <= max_shift and abs(dy) <= max_shift
