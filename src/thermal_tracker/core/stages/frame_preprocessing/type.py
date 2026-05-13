"""Типы атомарных операций предобработки кадра.

Модуль содержит enum со всеми атомарными операциями, которые можно
включать в стадию preprocessing. Enum применяется в конфигурациях и
пресетах: внешний код передаёт список значений в FramePreprocessorManager,
а менеджер по каждому значению создаёт соответствующую операцию.
"""

from __future__ import annotations

from enum import StrEnum


class FramePreprocessorType(StrEnum):
    """Доступные атомарные операции на стадии предобработки кадра."""

    RESIZE = "resize"  # Уменьшает размер всех каналов до заданной ширины с сохранением пропорций.
    GAUSSIAN_BLUR = "gaussian_blur"  # Сглаживает gray гауссовым фильтром.
    MEDIAN_BLUR = "median_blur"  # Сглаживает gray медианным фильтром.
    BILATERAL = "bilateral"  # Сглаживает gray bilateral-фильтром, сохраняя границы объектов.
    NORMALIZE_MINMAX = "minmax_normalize"  # Линейная нормализация gray -> normalized в диапазоне [0, 1].
    CLAHE_CONTRAST = "clahe_contrast"  # Усиливает локальный контраст в normalized через CLAHE.
    PERCENTILE_NORMALIZE = "percentile_normalize"  # Перцентильная нормализация gray -> normalized.
    GRADIENT = "gradient"  # Считает карту градиентов из normalized в gradient.
    SHARPNESS_METRIC = "sharpness_metric"  # Заполняет ProcessedFrame.quality.sharpness.
