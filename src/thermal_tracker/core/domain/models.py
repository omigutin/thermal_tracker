"""Базовые модели данных для пайплайна трекинга."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TrackerState(str, Enum):
    """Состояния жизненного цикла одной отслеживаемой цели.

    - IDLE       — цель не выбрана. Стартовое состояние и состояние после полного сброса.
    - TRACKING   — цель уверенно сопровождается. Модель движения обновляется обычным путём.
    - SEARCHING  — цель временно потеряна. Модель движения только предсказывает прогноз
                   и не обновляется, пока не появится подтверждённый кандидат.
    - RECOVERING — найден кандидат на восстановление цели, но он ещё не подтверждён историей.
                   Модель движения не обновляется до подтверждения, чтобы плохой кандидат
                   не испортил траекторию.
    - LOST       — цель окончательно потеряна. Модель движения недостоверна. Дальше нужен
                   новый клик пользователя или внешний сброс.
    """

    IDLE = "IDLE"
    TRACKING = "TRACKING"
    SEARCHING = "SEARCHING"
    RECOVERING = "RECOVERING"
    LOST = "LOST"


@dataclass(frozen=True)
class BoundingBox:
    """Прямоугольник в координатах кадра."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    def pad(self, pad_x: int, pad_y: int) -> "BoundingBox":
        return BoundingBox(
            x=self.x - pad_x,
            y=self.y - pad_y,
            width=self.width + pad_x * 2,
            height=self.height + pad_y * 2,
        )

    def clamp(self, frame_shape: tuple[int, int] | tuple[int, int, int]) -> "BoundingBox":
        frame_h, frame_w = frame_shape[:2]
        x1 = min(max(0, self.x), max(0, frame_w - 1))
        y1 = min(max(0, self.y), max(0, frame_h - 1))
        x2 = min(max(x1 + 1, self.x2), frame_w)
        y2 = min(max(y1 + 1, self.y2), frame_h)
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def to_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def intersection_over_union(self, other: "BoundingBox") -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union

    @classmethod
    def from_center(cls, cx: float, cy: float, width: int, height: int) -> "BoundingBox":
        return cls(
            x=int(round(cx - width / 2.0)),
            y=int(round(cy - height / 2.0)),
            width=int(round(width)),
            height=int(round(height)),
        )


@dataclass
class FrameQuality:
    """Метрики качества одного кадра, заполняемые атомарными операциями preprocessing.

    sharpness — оценка резкости центральной части кадра (Laplacian-метрика).
    blurred   — флаг, что резкость существенно ниже базовой.
    """

    sharpness: float = 0.0
    blurred: bool = False


@dataclass
class ProcessedFrame:
    """Кадр после предобработки."""

    bgr: np.ndarray
    gray: np.ndarray
    normalized: np.ndarray
    gradient: np.ndarray
    quality: FrameQuality | None = None
