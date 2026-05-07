"""Подготовка теплового кадра перед трекингом.

Здесь не пытаемся быть очень умными:
- слегка чистим шум;
- выравниваем контраст;
- считаем карту градиентов.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...config import PreprocessingConfig
from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor


def _make_odd(value: int) -> int:
    """Делает размер фильтра нечётным."""

    return value if value % 2 == 1 else value + 1


class ThermalFramePreprocessor(BaseFramePreprocessor):
    """Готовит кадр так, чтобы трекеру было проще жить."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Прогоняет один кадр через базовую подготовку."""

        display_frame = self._resize_if_needed(frame)
        gray = self._to_gray(display_frame)

        gaussian_kernel = _make_odd(max(1, self.config.gaussian_kernel))
        if gaussian_kernel > 1:
            gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

        median_kernel = _make_odd(max(1, self.config.median_kernel))
        if median_kernel > 1:
            gray = cv2.medianBlur(gray, median_kernel)

        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        normalized = self._clahe.apply(normalized)

        gradient_input = normalized
        gradient_blur = _make_odd(max(1, self.config.gradient_blur_kernel))
        if gradient_blur > 1:
            gradient_input = cv2.GaussianBlur(gradient_input, (gradient_blur, gradient_blur), 0)

        grad_x = cv2.Sobel(gradient_input, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gradient_input, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return ProcessedFrame(
            bgr=display_frame,
            gray=gray,
            normalized=normalized,
            gradient=gradient,
        )

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Уменьшает слишком широкие кадры, чтобы интерфейс не страдал."""

        target_width = self.config.resize_width
        if target_width is None or frame.shape[1] <= target_width:
            return frame.copy()

        scale = target_width / frame.shape[1]
        target_height = int(round(frame.shape[0] * scale))
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Переводит кадр в градации серого, если это ещё не так."""

        if frame.ndim == 2:
            return frame.copy()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
