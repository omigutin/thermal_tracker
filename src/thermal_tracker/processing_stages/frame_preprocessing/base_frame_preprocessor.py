"""Базовый класс для предобработки кадра."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...domain.models import ProcessedFrame


class BaseFramePreprocessor(ABC):
    """Любая предобработка должна превращать сырой кадр в `ProcessedFrame`."""

    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Возвращает подготовленный кадр с нужными представлениями."""
