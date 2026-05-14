from __future__ import annotations

from enum import StrEnum


class FrameStabilizerType(StrEnum):
    """Типы операций стабилизации кадра."""
    PHASE_CORRELATION = "phase_correlation"  # Быстро оценивает общий сдвиг кадра через phase correlation.
    FEATURE_AFFINE = "feature_affine"  # Оценивает аффинный сдвиг по ключевым точкам OpenCV.
