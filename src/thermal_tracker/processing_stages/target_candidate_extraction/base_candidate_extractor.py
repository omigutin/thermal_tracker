"""Базовый класс для сборки объектов из детекта."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import DetectedObject, MotionDetectionResult, ProcessedFrame


class BaseObjectBuilder(ABC):
    """Преобразует сырую маску или отклик детектора в список объектов."""

    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def build(self, frame: ProcessedFrame, detection: MotionDetectionResult) -> list[DetectedObject]:
        """Возвращает список найденных объектов."""
