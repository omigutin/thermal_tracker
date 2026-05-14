from __future__ import annotations

import cv2
import numpy as np

from ....domain.models import BoundingBox
from .base_target_predictor import BaseTargetPredictor


class KalmanTargetPredictor(BaseTargetPredictor):
    """Прогнозирует bbox цели через фильтр Калмана."""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1, min_size: int = 1) -> None:
        """Создать прогнозатор с фильтром Калмана."""
        self._process_noise: float = process_noise
        self._measurement_noise: float = measurement_noise
        self._min_size: int = min_size
        self._filter: cv2.KalmanFilter = self._create_filter()
        self._bbox: BoundingBox | None = None

    def reset(self) -> None:
        """Сбросить внутреннее состояние прогнозатора."""
        self._filter = self._create_filter()
        self._bbox = None

    def initialize(self, bbox: BoundingBox) -> None:
        """Инициализировать прогнозатор первым bbox цели."""
        state = np.array(
            [
                [bbox.x],
                [bbox.y],
                [bbox.width],
                [bbox.height],
                [0],
                [0],
                [0],
                [0],
            ],
            dtype=np.float32,
        )

        self._filter.statePost = state.copy()
        self._filter.statePre = state.copy()
        self._bbox = bbox

    def predict(self) -> BoundingBox | None:
        """Вернуть прогноз следующего положения цели."""
        if self._bbox is None:
            return None

        prediction = self._filter.predict()

        return BoundingBox(
            x=int(round(float(prediction[0, 0]))),
            y=int(round(float(prediction[1, 0]))),
            width=max(self._min_size, int(round(float(prediction[2, 0])))),
            height=max(self._min_size, int(round(float(prediction[3, 0])))),
        )

    def update(self, bbox: BoundingBox) -> None:
        """Обновить прогнозатор новым bbox цели."""
        if self._bbox is None:
            self.initialize(bbox)
            return

        measurement = np.array(
            [
                [bbox.x],
                [bbox.y],
                [bbox.width],
                [bbox.height],
            ],
            dtype=np.float32,
        )

        self._filter.correct(measurement)
        self._bbox = bbox

    def _create_filter(self) -> cv2.KalmanFilter:
        """Создать и настроить фильтр Калмана."""
        kalman_filter = cv2.KalmanFilter(8, 4)

        kalman_filter.transitionMatrix = np.array(
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

        kalman_filter.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        kalman_filter.processNoiseCov = (np.eye(8, dtype=np.float32) * self._process_noise)
        kalman_filter.measurementNoiseCov = (np.eye(4, dtype=np.float32) * self._measurement_noise)

        return kalman_filter
