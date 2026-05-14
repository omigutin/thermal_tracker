from __future__ import annotations

from enum import StrEnum


class FramePreprocessorType(StrEnum):
    """Типы операций предобработки кадра."""

    # Уменьшает все каналы кадра до заданной ширины с сохранением пропорций.
    RESIZE = "resize"

    # Сглаживает gray-канал гауссовым фильтром.
    GAUSSIAN_BLUR = "gaussian_blur"

    # Сглаживает gray-канал медианным фильтром.
    MEDIAN_BLUR = "median_blur"

    # Сглаживает gray-канал bilateral-фильтром с сохранением границ.
    BILATERAL_BLUR = "bilateral_blur"

    # Нормализует gray-канал в normalized через min-max растяжение.
    MINMAX_NORMALIZE = "minmax_normalize"

    # Нормализует gray-канал в normalized через отсечение перцентилей.
    PERCENTILE_NORMALIZE = "percentile_normalize"

    # Усиливает локальный контраст normalized-канала через CLAHE.
    CLAHE_CONTRAST = "clahe_contrast"

    # Считает карту градиентов из normalized-канала.
    GRADIENT = "gradient"

    # Считает метрику резкости кадра и записывает её в quality.
    SHARPNESS_METRIC = "sharpness_metric"
