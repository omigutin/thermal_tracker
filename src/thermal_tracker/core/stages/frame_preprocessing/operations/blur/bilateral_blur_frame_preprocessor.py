"""Атомарная операция: bilateral-сглаживание серого канала.

Сглаживает шум, стараясь сохранить границы объектов. Применяется к gray.
"""

from __future__ import annotations

from dataclasses import replace

import cv2

from .....domain.models import ProcessedFrame
from ..base_frame_preprocessor import BaseFramePreprocessor


class BilateralFramePreprocessor(BaseFramePreprocessor):
    """Применяет bilateral-фильтр к gray-каналу."""

    def __init__(
        self,
        diameter: int = 7,
        sigma_color: float = 40.0,
        sigma_space: float = 40.0,
    ) -> None:
        self._diameter = diameter
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        gray = cv2.bilateralFilter(
            frame.gray,
            self._diameter,
            self._sigma_color,
            self._sigma_space,
        )
        return replace(frame, gray=gray)
