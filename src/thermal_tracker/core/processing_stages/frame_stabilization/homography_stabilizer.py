"""Заготовка под стабилизацию через гомографию.

Это уже более тяжёлая артиллерия:
- потенциально полезна при сложной геометрии;
- требует больше вычислений и хороших точек;
- для текущего этапа это скорее резерв на будущее.
"""

from __future__ import annotations

from .base_stabilizer import BaseMotionEstimator


class HomographyMotionEstimator(BaseMotionEstimator):
    """Будущая стабилизация через оценку гомографии."""

    def estimate(self, frame):
        raise NotImplementedError("Стабилизация через гомографию пока не реализована.")
