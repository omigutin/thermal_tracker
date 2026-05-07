"""Базовый контракт интерфейса к нейросетевой модели.

Интерфейс скрывает конкретную библиотеку инференса и возвращает результат
в доменных структурах проекта. Позже сюда можно подключить внешнюю общую
библиотеку интерфейсов без переписывания сценариев и трекеров.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..domain.models import DetectedObject


class BaseNnetInterface(ABC):
    """Общий интерфейс для нейросетевого детектора или трекера."""

    interface_name = "base"
    is_ready = False

    @abstractmethod
    def track(self, frame: np.ndarray) -> list[DetectedObject]:
        """Обрабатывает кадр и возвращает найденные объекты."""
