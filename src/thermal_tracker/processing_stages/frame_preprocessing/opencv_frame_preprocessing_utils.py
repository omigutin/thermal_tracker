"""Общие маленькие помощники для разных препроцессоров."""

from __future__ import annotations

import cv2
import numpy as np


def make_odd(value: int) -> int:
    """Делает размер ядра нечётным, чтобы OpenCV не закатил истерику."""

    return value if value % 2 == 1 else value + 1


def resize_if_needed(frame: np.ndarray, target_width: int | None) -> np.ndarray:
    """Уменьшает слишком широкий кадр до нужной ширины."""

    if target_width is None or frame.shape[1] <= target_width:
        return frame.copy()

    scale = target_width / frame.shape[1]
    target_height = int(round(frame.shape[0] * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Переводит кадр в градации серого."""

    if frame.ndim == 2:
        return frame.copy()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def build_gradient(gray: np.ndarray, blur_kernel: int = 3) -> np.ndarray:
    """Строит простую карту градиентов."""

    work = gray
    blur_kernel = make_odd(max(1, blur_kernel))
    if blur_kernel > 1:
        work = cv2.GaussianBlur(work, (blur_kernel, blur_kernel), 0)

    grad_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    return cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def normalize_minmax(gray: np.ndarray) -> np.ndarray:
    """Простая нормализация яркости в диапазон 0..255."""

    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
