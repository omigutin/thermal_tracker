"""
    Типы препроцессоров кадра.
    Модуль содержит enum со всеми препроцессорами,
    которые могут использоваться на стадии предварительной обработки кадра.
    Enum применяется в конфигурациях и пресетах:
    внешний код может передать тип препроцессора в FramePreprocessorManager,
    а менеджер преобразует его в готовый экземпляр соответствующего препроцессора.
"""

from __future__ import annotations

from enum import StrEnum


class FramePreprocessorType(StrEnum):
    """ Доступные типы препроцессоров кадра. """

    IDENTITY = "identity"  # Возвращает кадр без смысловой обработки, только в общем формате ProcessedFrame.
    THERMAL = "thermal"  # Базовая тепловизионная подготовка: grayscale, нормализация, сглаживание, градиент.
    BILATERAL = "bilateral"  # Подавляет шум, стараясь сохранить границы объектов.
    CLAHE_CONTRAST = "clahe_contrast"  # Усиливает локальный контраст через CLAHE.
    PERCENTILE_NORMALIZE = "percentile_normalize"  # Нормализует яркость по процентилям, отрезая выбросы.

    # AGC_COMPENSATION = "agc_compensation"  # Компенсация авто-подстройки яркости тепловизора.
    # GRADIENT_ENHANCED = "gradient_enhanced"  # Подчёркивание границ и мелких контрастных деталей.
    # TEMPORAL_DENOISE = "temporal_denoise"  # Подавление шума с учетом соседних кадров.
