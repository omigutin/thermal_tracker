"""Контрольный режим без стабилизации.

Нужен обязательно:
- чтобы сравнивать, помогает ли компенсация движения вообще;
- чтобы не путать эффект нового трекера с эффектом нового стабилизатора.
"""

from __future__ import annotations

from ...domain.models import GlobalMotion, ProcessedFrame
from .base_stabilizer import BaseMotionEstimator


class NoMotionEstimator(BaseMotionEstimator):
    """Всегда говорит, что камера никуда не двигалась."""

    implementation_name = "none"
    is_ready = True

    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        return GlobalMotion()
