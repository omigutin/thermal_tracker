"""Простая модель движения с постоянной скоростью."""

from __future__ import annotations

from ....domain.models import BoundingBox
from .base_motion_model import BaseMotionModel


class ConstantVelocityMotionModel(BaseMotionModel):
    """Держит последний bbox и примитивную скорость его центра."""

    implementation_name = "constant_velocity"
    is_ready = True

    def __init__(self) -> None:
        self._bbox: BoundingBox | None = None
        self._vx = 0.0
        self._vy = 0.0

    def reset(self) -> None:
        self._bbox = None
        self._vx = 0.0
        self._vy = 0.0

    def initialize(self, bbox: BoundingBox) -> None:
        self._bbox = bbox
        self._vx = 0.0
        self._vy = 0.0

    def predict(self) -> BoundingBox | None:
        if self._bbox is None:
            return None
        cx, cy = self._bbox.center
        return BoundingBox.from_center(cx + self._vx, cy + self._vy, self._bbox.width, self._bbox.height)

    def update(self, bbox: BoundingBox) -> None:
        if self._bbox is None:
            self.initialize(bbox)
            return
        old_cx, old_cy = self._bbox.center
        new_cx, new_cy = bbox.center
        self._vx = new_cx - old_cx
        self._vy = new_cy - old_cy
        self._bbox = bbox
