from __future__ import annotations

from ....domain.models import BoundingBox
from .base_target_predictor import BaseTargetPredictor


class ConstantVelocityTargetPredictor(BaseTargetPredictor):
    """Прогнозирует положение цели по последней скорости центра bbox."""

    def __init__(self) -> None:
        """Создать пустой прогнозатор без начального измерения."""
        self._bbox: BoundingBox | None = None
        self._vx: float = 0.0
        self._vy: float = 0.0

    def reset(self) -> None:
        """Сбросить внутреннее состояние прогнозатора."""
        self._bbox = None
        self._vx = 0.0
        self._vy = 0.0

    def initialize(self, bbox: BoundingBox) -> None:
        """Инициализировать прогнозатор первым bbox цели."""
        self._bbox = bbox
        self._vx = 0.0
        self._vy = 0.0

    def predict(self) -> BoundingBox | None:
        """Вернуть прогноз следующего положения цели."""
        if self._bbox is None:
            return None

        center_x, center_y = self._bbox.center

        return BoundingBox.from_center(
            center_x + self._vx,
            center_y + self._vy,
            self._bbox.width,
            self._bbox.height,
        )

    def update(self, bbox: BoundingBox) -> None:
        """Обновить прогнозатор новым bbox цели."""
        if self._bbox is None:
            self.initialize(bbox)
            return

        old_center_x, old_center_y = self._bbox.center
        new_center_x, new_center_y = bbox.center

        self._vx = new_center_x - old_center_x
        self._vy = new_center_y - old_center_y
        self._bbox = bbox
