"""Базовый контракт для движков инференса.

Движок знает, как запустить конкретную модель и вернуть детекции
в наших доменных структурах. Остальной код не должен разбираться
во внутренностях Ultralytics, Torch и прочих любителей сюрпризов.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...domain.models import DetectedObject


class BaseInferenceEngine(ABC):
    """Общий интерфейс для нейросетевого детектора/трекера."""

    engine_name = "base"
    is_ready = False

    @abstractmethod
    def track(self, frame: np.ndarray) -> list[DetectedObject]:
        """Обрабатывает кадр и возвращает найденные объекты."""

