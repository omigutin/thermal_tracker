from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class MotionLocalizerResult:
    """
        Результат локализации движения на кадре.
        Хранит финальную бинарную маску движения и дополнительную карту интенсивности
        движения, если конкретный алгоритм может её вернуть.
        Ожидаемый формат mask:
        - двумерный массив;
        - размер совпадает с обрабатываемым кадром;
        - 0 — движения нет;
        - ненулевое значение — движение есть.
    """

    mask: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    motion_score: float = 0.0

    def __post_init__(self) -> None:
        """Проверить базовую согласованность результата."""
        if self.mask.ndim != 2:
            raise ValueError("mask must be a 2D array.")

        if self.confidence_map is not None and self.confidence_map.shape != self.mask.shape:
            raise ValueError("confidence_map shape must match mask shape.")

        if not 0.0 <= self.motion_score <= 1.0:
            raise ValueError("motion_score must be in range [0.0, 1.0].")

    @property
    def has_motion(self) -> bool:
        """Проверить, есть ли движение в финальной маске."""
        return bool(np.count_nonzero(self.mask))

    @classmethod
    def empty_like(cls, frame: np.ndarray) -> MotionLocalizerResult:
        """Создать пустой результат локализации движения по размеру кадра."""
        empty = np.zeros_like(frame, dtype=np.uint8)

        return cls(mask=empty, confidence_map=empty.copy(), motion_score=0.0)
