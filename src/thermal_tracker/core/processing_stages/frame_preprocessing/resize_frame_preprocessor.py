"""Атомарная операция: масштабирование всех каналов кадра до целевой ширины."""

from __future__ import annotations

import cv2

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor


class ResizeFramePreprocessor(BaseFramePreprocessor):
    """Уменьшает кадр до заданной ширины с сохранением пропорций.

    Если текущая ширина уже не превышает целевую, кадр возвращается без
    изменений. Если ``target_width`` равен ``None``, операция бездействует.
    """

    def __init__(self, target_width: int | None) -> None:
        self._target_width = target_width

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        target_width = self._target_width
        if target_width is None or frame.bgr.shape[1] <= target_width:
            return frame

        scale = target_width / frame.bgr.shape[1]
        target_height = int(round(frame.bgr.shape[0] * scale))
        size = (target_width, target_height)
        return ProcessedFrame(
            bgr=cv2.resize(frame.bgr, size, interpolation=cv2.INTER_AREA),
            gray=cv2.resize(frame.gray, size, interpolation=cv2.INTER_AREA),
            normalized=cv2.resize(frame.normalized, size, interpolation=cv2.INTER_AREA),
            gradient=cv2.resize(frame.gradient, size, interpolation=cv2.INTER_AREA),
            quality=frame.quality,
        )
