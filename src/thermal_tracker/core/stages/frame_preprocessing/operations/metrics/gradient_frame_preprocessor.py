"""Атомарная операция: расчёт карты градиентов из normalized в gradient."""

from __future__ import annotations

from dataclasses import replace

from .....domain.models import ProcessedFrame
from ..base_frame_preprocessor import BaseFramePreprocessor
from ...opencv_frame_preprocessing_utils import build_gradient


class GradientFramePreprocessor(BaseFramePreprocessor):
    """Считает карту градиентов из normalized и записывает её в gradient."""

    def __init__(self, blur_kernel: int) -> None:
        self._blur_kernel = max(1, blur_kernel)

    def process(self, frame: ProcessedFrame) -> ProcessedFrame:
        gradient = build_gradient(frame.normalized, blur_kernel=self._blur_kernel)
        return replace(frame, gradient=gradient)
