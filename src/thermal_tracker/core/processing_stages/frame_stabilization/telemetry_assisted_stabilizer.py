"""Заготовка под стабилизацию с телеметрией платформы.

Это тот самый взрослый вариант на будущее:
- можно использовать углы, IMU, энкодеры или данные крутилки;
- визуальная оценка движения тогда становится не единственным источником.
"""

from __future__ import annotations

from .base_stabilizer import BaseMotionEstimator


class TelemetryAssistedMotionEstimator(BaseMotionEstimator):
    """Будущая стабилизация с учётом внешней телеметрии."""

    implementation_name = "telemetry_assisted_stabilizer"
    is_ready = False

    def estimate(self, frame):
        raise NotImplementedError("Стабилизация с телеметрией пока не реализована.")
