"""Заготовка под ECC-аффинное выравнивание.

Подходит, когда сдвига уже мало и хочется учесть небольшой поворот
или лёгкое изменение масштаба.
"""

from __future__ import annotations

from .base_stabilizer import BaseMotionEstimator


class EccAffineMotionEstimator(BaseMotionEstimator):
    """Будущая аффинная стабилизация на основе ECC."""

    def estimate(self, frame):
        raise NotImplementedError("ECC affine пока не реализован.")
