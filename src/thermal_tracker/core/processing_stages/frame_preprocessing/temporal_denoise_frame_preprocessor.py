"""Заготовка под временное подавление шума.

Идея полезная для тепловизора: смотреть не только на текущий кадр,
но и на пару соседних, чтобы уменьшать дрожащий шум.

Пока не реализовано, потому что для этого надо аккуратно хранить историю
и не испортить задержку живого GUI.
"""

from __future__ import annotations

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor


class TemporalDenoisePreprocessor(BaseFramePreprocessor):
    """Будущий препроцессор с накоплением нескольких кадров."""

    def process(self, frame) -> ProcessedFrame:
        raise NotImplementedError("Временное подавление шума пока не реализовано.")
