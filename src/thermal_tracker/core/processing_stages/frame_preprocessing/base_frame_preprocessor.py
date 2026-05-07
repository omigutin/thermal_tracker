"""
    Базовый интерфейс препроцессора кадра.
    Модуль содержит абстрактный класс для всех препроцессоров, которые преобразуют
    сырой входной кадр в ProcessedFrame.
    Каждый препроцессор должен подготовить основные представления кадра:
        - bgr: кадр для отображения и дальнейшей работы с координатами;
        - gray: одноканальное представление;
        - normalized: нормализованное представление яркости;
        - gradient: карта градиентов или контурной структуры.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...domain.models import ProcessedFrame


class BaseFramePreprocessor(ABC):
    """Базовый интерфейс атомарного препроцессора кадра."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Преобразовать сырой кадр в подготовленный ProcessedFrame."""
        pass