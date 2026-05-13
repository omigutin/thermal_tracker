"""Атомарная операция: измерение резкости центральной части кадра.

Метрика повторяет логику ``_measure_frame_sharpness`` в текущем трекере и
кладёт результат в ``ProcessedFrame.quality.sharpness``. На последующих
этапах (адаптивный Калман, blur-hold) трекер будет читать готовое значение
вместо того, чтобы считать его внутри себя.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .....domain.models import FrameQuality, ProcessedFrame
from ..base_frame_preprocessor import BaseFramePreprocessor


class SharpnessMetricFramePreprocessor(BaseFramePreprocessor):
    """Заполняет ProcessedFrame.quality.sharpness Laplacian-метрикой."""

    def __init__(
        self,
        crop_left: float = 0.08,
        crop_right: float = 0.92,
        crop_top: float = 0.18,
        crop_bottom: float = 0.82,
        percentile: float = 90.0,
    ) -> None:
        self._crop_left = crop_left
        self._crop_right = crop_right
        self._crop_top = crop_top
        self._crop_bottom = crop_bottom
        self._percentile = percentile

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        gray = frame.gray
        frame_h, frame_w = gray.shape[:2]
        x1 = int(round(frame_w * self._crop_left))
        x2 = int(round(frame_w * self._crop_right))
        y1 = int(round(frame_h * self._crop_top))
        y2 = int(round(frame_h * self._crop_bottom))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            roi = gray

        laplacian = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
        sharpness = float(np.percentile(np.abs(laplacian), self._percentile))

        previous = frame.quality
        if previous is None:
            quality = FrameQuality(sharpness=sharpness, blurred=False)
        else:
            quality = replace(previous, sharpness=sharpness)
        return replace(frame, quality=quality)
