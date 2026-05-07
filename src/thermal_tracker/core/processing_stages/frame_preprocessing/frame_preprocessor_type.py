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

    IDENTITY = "identity"
    THERMAL = "thermal"
    BILATERAL = "bilateral"
    CLAHE_CONTRAST = "clahe_contrast"
    PERCENTILE_NORMALIZE = "percentile_normalize"

    # AGC_COMPENSATION = "agc_compensation"
    # GRADIENT_ENHANCED = "gradient_enhanced"
    # TEMPORAL_DENOISE = "temporal_denoise"