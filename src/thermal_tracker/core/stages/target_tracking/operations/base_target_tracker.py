from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import BoundingBox, ProcessedFrame
from ..result import TargetTrackingResult
from ...frame_stabilization import FrameStabilizerResult


class BaseTargetTracker(ABC):
    """Базовый интерфейс трекера одной цели."""

    @abstractmethod
    def snapshot(self, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Вернуть текущее состояние трекера."""
        raise NotImplementedError

    @abstractmethod
    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TargetTrackingResult:
        """Начать сопровождение цели по точке выбора."""
        raise NotImplementedError

    @abstractmethod
    def update(self, frame: ProcessedFrame, motion: FrameStabilizerResult) -> TargetTrackingResult:
        """Обновить состояние трекера по новому кадру."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> TargetTrackingResult:
        """Сбросить текущее состояние трекера."""
        raise NotImplementedError

    def resume_tracking(self, frame: ProcessedFrame, bbox: BoundingBox, track_id: int) -> TargetTrackingResult:
        """
            Возобновить сопровождение цели с заданными bbox и track_id.
            Используется после восстановления цели, чтобы продолжить старый трек, а не создавать новый идентификатор.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support target tracking resume.")
