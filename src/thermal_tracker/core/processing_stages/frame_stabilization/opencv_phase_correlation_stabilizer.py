"""Оценка грубого движения камеры между соседними кадрами."""

from __future__ import annotations

import cv2

from ...config import GlobalMotionConfig
from ...domain.models import GlobalMotion, ProcessedFrame
from .base_stabilizer import BaseMotionEstimator


def _make_odd(value: int) -> int:
    """Делает размер ядра нечётным, потому что OpenCV это любит."""

    return value if value % 2 == 1 else value + 1


class PhaseCorrelationMotionEstimator(BaseMotionEstimator):
    """Оценивает глобальный сдвиг кадра через фазовую корреляцию.

    Это не волшебство и не полноценная стабилизация, а быстрый способ
    понять, насколько вся картинка в среднем уехала вправо, влево,
    вверх или вниз.
    """

    def __init__(self, config: GlobalMotionConfig) -> None:
        self.config = config
        self._previous = None
        self._window = None

    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        """Возвращает оценку движения камеры для текущего кадра."""

        if not self.config.enabled:
            return GlobalMotion()

        current = frame.normalized.astype("float32") / 255.0
        scale = self.config.downscale
        if scale != 1.0:
            current = cv2.resize(current, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        blur_kernel = _make_odd(max(1, self.config.blur_kernel))
        if blur_kernel > 1:
            current = cv2.GaussianBlur(current, (blur_kernel, blur_kernel), 0)

        if self._window is None or self._window.shape != current.shape:
            self._window = cv2.createHanningWindow((current.shape[1], current.shape[0]), cv2.CV_32F)

        if self._previous is None:
            self._previous = current
            return GlobalMotion()

        (dx, dy), response = cv2.phaseCorrelate(self._previous, current, self._window)
        self._previous = current

        max_shift = max(frame.bgr.shape[:2]) * self.config.max_shift_ratio
        scaled_dx = dx / scale
        scaled_dy = dy / scale
        valid = (
            response >= self.config.min_response
            and abs(scaled_dx) <= max_shift
            and abs(scaled_dy) <= max_shift
        )

        if not valid:
            return GlobalMotion(response=response, valid=False)

        return GlobalMotion(dx=scaled_dx, dy=scaled_dy, response=response, valid=True)
