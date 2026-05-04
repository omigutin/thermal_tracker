"""Заготовка под препроцессор с усилением контуров.

Такой вариант полезен, когда трекеру важнее форма и границы объекта,
чем абсолютные температуры пикселей.
"""

from __future__ import annotations

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor


class GradientEnhancedPreprocessor(BaseFramePreprocessor):
    """Будущий препроцессор с явным акцентом на контурную структуру."""

    implementation_name = "gradient_enhanced"
    is_ready = False

    def process(self, frame) -> ProcessedFrame:
        raise NotImplementedError("Препроцессор с усилением градиентов пока не реализован.")
