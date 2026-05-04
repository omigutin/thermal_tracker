"""Исключения доменного уровня для thermal_tracker."""

from __future__ import annotations


class MotionDetectionError(Exception):
    """Базовая ошибка подсистемы трекинга."""


class VideoOpenError(MotionDetectionError):
    """Не удалось открыть видео или источник кадров."""

