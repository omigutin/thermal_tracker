"""Базовый класс для single-target трекеров."""

from __future__ import annotations

from abc import ABC, abstractmethod

from thermal_tracker.core.domain.models import BoundingBox, ProcessedFrame, TrackSnapshot
from thermal_tracker.core.stages.frame_stabilization.result import FrameStabilizerResult


class BaseSingleTargetTracker(ABC):
    """Любой трекер одной цели должен уметь стартовать, обновляться и сбрасываться."""

    @abstractmethod
    def snapshot(self, motion: FrameStabilizerResult) -> TrackSnapshot:
        """Возвращает текущее состояние трекера."""

    @abstractmethod
    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает сопровождение по одному клику."""

    @abstractmethod
    def update(self, frame: ProcessedFrame, motion: FrameStabilizerResult) -> TrackSnapshot:
        """Обновляет состояние на следующем кадре."""

    @abstractmethod
    def reset(self) -> TrackSnapshot:
        """Сбрасывает текущее состояние трекера."""

    def resume_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        track_id: int,
    ) -> TrackSnapshot:
        """Возобновляет сопровождение цели с конкретным track_id и bbox.

        Используется pipeline-ом после подтверждённого recovery, чтобы
        продолжить трек с тем же ID и не плодить новые. По умолчанию
        метод не реализован: конкретный трекер должен переопределить
        его, если участвует в сценариях с recovery.
        """

        raise NotImplementedError("resume_tracking is not implemented for this tracker.")
