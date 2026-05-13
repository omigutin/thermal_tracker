"""Типы методов оценки движения камеры."""

from __future__ import annotations

from enum import StrEnum


class FrameStabilizerType(StrEnum):
    """Доступные методы стадии frame_stabilization."""
    NO = "no"  # Отключает оценку движения камеры.
    OPENCV_PHASE_CORRELATION = "opencv_phase_correlation"  # Быстро оценивает общий сдвиг кадра через phase correlation.
    OPENCV_FEATURE_AFFINE = "opencv_feature_affine"  # Оценивает аффинный сдвиг по ключевым точкам OpenCV.
