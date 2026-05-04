"""Заготовка под ECC-оценку чистого сдвига.

ECC часто даёт более аккуратное выравнивание, чем совсем грубые методы,
но обычно тяжелее по вычислениям и капризнее к начальному приближению.
"""

from __future__ import annotations

from .base_stabilizer import BaseMotionEstimator


class EccTranslationMotionEstimator(BaseMotionEstimator):
    """Будущая реализация ECC для режима только-сдвиг."""

    implementation_name = "ecc_translation_stabilizer"
    is_ready = False

    def estimate(self, frame):
        raise NotImplementedError("ECC translation пока не реализован.")
