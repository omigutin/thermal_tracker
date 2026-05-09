"""Атомарная операция: min-max-нормализация яркости gray в normalized."""

from __future__ import annotations

from dataclasses import replace

import cv2

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor


class NormalizeMinMaxFramePreprocessor(BaseFramePreprocessor):
    """Линейно растягивает яркость gray в диапазон [0, 255] и пишет в normalized."""

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        normalized = cv2.normalize(frame.gray, None, 0, 255, cv2.NORM_MINMAX)
        return replace(frame, normalized=normalized)
