from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import BoundingBox


class BaseTargetPredictor(ABC):
    """Базовый интерфейс прогнозатора положения цели."""

    @abstractmethod
    def reset(self) -> None:
        """Сбросить внутреннее состояние прогнозатора."""
        raise NotImplementedError

    @abstractmethod
    def initialize(self, bbox: BoundingBox) -> None:
        """Инициализировать прогнозатор первым измерением цели."""
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> BoundingBox | None:
        """Вернуть прогноз следующего положения цели."""
        raise NotImplementedError

    @abstractmethod
    def update(self, bbox: BoundingBox) -> None:
        """Обновить прогнозатор новым измерением положения цели."""
        raise NotImplementedError
