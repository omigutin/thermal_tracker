"""Контрольная модель движения без прогноза."""

from __future__ import annotations

from ....domain.models import BoundingBox
from .base_motion_model import BaseMotionModel


class NoMotionModel(BaseMotionModel):
    """Просто помнит последний bbox и ничего не предсказывает умнее этого."""

    def __init__(self) -> None:
        self._bbox: BoundingBox | None = None

    def reset(self) -> None:
        self._bbox = None

    def initialize(self, bbox: BoundingBox) -> None:
        self._bbox = bbox

    def predict(self) -> BoundingBox | None:
        return self._bbox

    def update(self, bbox: BoundingBox) -> None:
        self._bbox = bbox
