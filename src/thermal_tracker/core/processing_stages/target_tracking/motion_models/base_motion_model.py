"""Базовый класс для моделей движения цели."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import BoundingBox


class BaseMotionModel(ABC):
    """Модель движения не трекает цель сама по себе, а только помогает прогнозом."""

    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def reset(self) -> None:
        """Сбрасывает внутреннее состояние модели."""

    @abstractmethod
    def initialize(self, bbox: BoundingBox) -> None:
        """Инициализирует модель первым измерением."""

    @abstractmethod
    def predict(self) -> BoundingBox | None:
        """Возвращает прогноз положения объекта."""

    @abstractmethod
    def update(self, bbox: BoundingBox) -> None:
        """Обновляет модель новым измерением."""
