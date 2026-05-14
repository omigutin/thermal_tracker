from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ...domain.models import ProcessedFrame
from .config import FramePreprocessorConfig
from .factory import FramePreprocessorFactory
from .operations import BaseFramePreprocessor


class FramePreprocessorManager:
    """Управляет последовательной предобработкой сырого кадра."""

    def __init__(self, operations: Sequence[FramePreprocessorConfig]) -> None:
        """Создать менеджер и подготовить активные runtime-операции."""
        self._preprocessors: tuple[BaseFramePreprocessor, ...] = (
            FramePreprocessorFactory.build_many(operations)
        )

    @property
    def preprocessors(self) -> tuple[BaseFramePreprocessor, ...]:
        """Вернуть подготовленные runtime-операции предобработки."""
        return self._preprocessors

    def process(self, raw: np.ndarray) -> ProcessedFrame:
        """Преобразовать сырой кадр в ProcessedFrame и применить цепочку операций."""
        frame = self._build_initial_frame(raw)

        for preprocessor in self._preprocessors:
            frame = preprocessor.process(frame)

        return frame

    @staticmethod
    def _build_initial_frame(raw: np.ndarray) -> ProcessedFrame:
        """Создать начальный ProcessedFrame из сырого изображения."""
        if raw.ndim == 2:
            gray = raw.copy()
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3:
            bgr = raw.copy()
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                f"Raw frame must be 2D grayscale or 3D BGR image, got shape: {raw.shape!r}."
            )

        return ProcessedFrame(
            bgr=bgr,
            gray=gray,
            normalized=gray.copy(),
            gradient=np.zeros_like(gray, dtype=np.uint8),
            quality=None,
        )
