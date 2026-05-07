"""Базовый класс для single-target трекеров."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import GlobalMotion, ProcessedFrame, TrackSnapshot


class BaseSingleTargetTracker(ABC):
    """Любой трекер одной цели должен уметь стартовать, обновляться и сбрасываться."""

    @abstractmethod
    def snapshot(self, motion: GlobalMotion) -> TrackSnapshot:
        """Возвращает текущее состояние трекера."""

    @abstractmethod
    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает сопровождение по одному клику."""

    @abstractmethod
    def update(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        """Обновляет состояние на следующем кадре."""

    @abstractmethod
    def reset(self) -> TrackSnapshot:
        """Сбрасывает текущее состояние трекера."""
