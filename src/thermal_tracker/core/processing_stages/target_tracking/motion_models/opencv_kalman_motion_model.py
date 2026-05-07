"""Модель движения на базе фильтра Калмана."""

from __future__ import annotations

import cv2
import numpy as np

from ....domain.models import BoundingBox
from .base_motion_model import BaseMotionModel


class KalmanMotionModel(BaseMotionModel):
    """Хранит состояние `[x, y, w, h, vx, vy, vw, vh]`."""

    def __init__(self) -> None:
        self._filter = cv2.KalmanFilter(8, 4)
        self._filter.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self._filter.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self._filter.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self._filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self._bbox: BoundingBox | None = None

    def reset(self) -> None:
        self.__init__()

    def initialize(self, bbox: BoundingBox) -> None:
        state = np.array([[bbox.x], [bbox.y], [bbox.width], [bbox.height], [0], [0], [0], [0]], dtype=np.float32)
        self._filter.statePost = state.copy()
        self._filter.statePre = state.copy()
        self._bbox = bbox

    def predict(self) -> BoundingBox | None:
        if self._bbox is None:
            return None
        prediction = self._filter.predict()
        return BoundingBox(
            x=int(round(float(prediction[0, 0]))),
            y=int(round(float(prediction[1, 0]))),
            width=max(1, int(round(float(prediction[2, 0])))),
            height=max(1, int(round(float(prediction[3, 0])))),
        )

    def update(self, bbox: BoundingBox) -> None:
        if self._bbox is None:
            self.initialize(bbox)
            return
        measurement = np.array([[bbox.x], [bbox.y], [bbox.width], [bbox.height]], dtype=np.float32)
        self._filter.correct(measurement)
        self._bbox = bbox
