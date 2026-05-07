"""Простые контракты между этапами пайплайна.

Это не DI-фреймворк и не культ интерфейсов ради интерфейсов.
Здесь просто зафиксирован общий язык между стадиями, чтобы было понятно,
что именно каждая часть системы должна уметь делать.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import (
    BoundingBox,
    DetectedObject,
    GlobalMotion,
    MotionDetectionResult,
    ProcessedFrame,
    SelectionResult,
    TrackSnapshot,
)


class VideoSource(Protocol):
    """Источник сырых кадров."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Читает следующий кадр."""

    def close(self) -> None:
        """Освобождает ресурс источника."""


class FramePreprocessor(Protocol):
    """Подготавливает сырой кадр к следующим шагам."""

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Возвращает обработанный кадр."""


class GlobalMotionEstimator(Protocol):
    """Оценивает движение камеры."""

    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        """Возвращает глобальный сдвиг кадра."""


class ClickInitializer(Protocol):
    """Инициализирует цель по одному клику."""

    def select(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> SelectionResult:
        """Находит объект вокруг точки."""


class SingleTargetTracker(Protocol):
    """Ведёт одну выбранную цель."""

    def snapshot(self, motion: GlobalMotion) -> TrackSnapshot:
        """Возвращает текущий снимок состояния."""

    def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Запускает трек по клику."""

    def update(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        """Обновляет трек на новом кадре."""

    def reset(self) -> TrackSnapshot:
        """Сбрасывает текущую цель."""


class MotionDetector(Protocol):
    """Ищет движение на уже подготовленном кадре."""

    def detect(self, frame: ProcessedFrame, motion: GlobalMotion) -> MotionDetectionResult:
        """Возвращает маску и служебную информацию по движению."""


class ObjectBuilder(Protocol):
    """Собирает объекты из маски движения или другого детектора."""

    def build(self, frame: ProcessedFrame, detection: MotionDetectionResult) -> list[DetectedObject]:
        """Преобразует результат детектора в список объектов."""


class FalseTargetFilter(Protocol):
    """Фильтрует ложные цели после грубого детекта."""

    def filter(
        self,
        frame: ProcessedFrame,
        objects: list[DetectedObject],
        motion: GlobalMotion,
    ) -> list[DetectedObject]:
        """Возвращает только те объекты, которым доверяем."""


class Reacquirer(Protocol):
    """Пытается вернуть цель после потери."""

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
    ) -> BoundingBox | None:
        """Возвращает новый bbox или `None`, если цель не нашли."""


class FrameRenderer(Protocol):
    """Рисует полезный оверлей поверх кадра."""

    def __call__(
        self,
        frame: ProcessedFrame,
        snapshot: TrackSnapshot,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Возвращает готовый BGR-кадр для показа."""
