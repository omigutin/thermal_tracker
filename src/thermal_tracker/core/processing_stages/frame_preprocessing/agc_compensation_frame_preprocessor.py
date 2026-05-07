"""Заготовка под компенсацию плавающего автоконтраста.

В интернет-роликах и на некоторых тепловизорах AGC может гулять,
из-за чего яркость цели меняется не потому, что цель изменилась,
а потому что камера решила красиво перетянуть картинку.
"""

from __future__ import annotations

from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor


class AgcCompensationPreprocessor(BaseFramePreprocessor):
    """Будущий препроцессор против артефактов автоконтраста."""

    implementation_name = "agc_compensation"
    is_ready = False

    def process(self, frame) -> ProcessedFrame:
        raise NotImplementedError("Компенсация AGC пока не реализована.")
