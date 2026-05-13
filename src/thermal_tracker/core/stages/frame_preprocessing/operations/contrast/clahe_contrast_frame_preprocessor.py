"""Атомарная операция: усиление локального контраста через CLAHE.

Применяется к каналу normalized.
"""

from __future__ import annotations

from dataclasses import replace

import cv2

from .....domain.models import ProcessedFrame
from ..base_frame_preprocessor import BaseFramePreprocessor


class ClaheContrastPreprocessor(BaseFramePreprocessor):
    """Усиливает локальный контраст в normalized через CLAHE."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        normalized = self._clahe.apply(frame.normalized)
        return replace(frame, normalized=normalized)
