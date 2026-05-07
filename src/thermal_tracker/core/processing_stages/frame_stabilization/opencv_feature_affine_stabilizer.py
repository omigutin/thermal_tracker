"""Оценка движения камеры по опорным точкам и аффинному преобразованию.

Это следующий шаг после phase correlation:
- умеет переживать не только чистый сдвиг;
- полезен, если есть лёгкий поворот или изменение масштаба;
- всё ещё остаётся относительно практичным для OpenCV-стека.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...domain.models import GlobalMotion, ProcessedFrame
from .base_stabilizer import BaseMotionEstimator


class FeatureAffineMotionEstimator(BaseMotionEstimator):
    """Оценивает глобальное движение по набору хороших точек."""

    def __init__(
        self,
        max_corners: int = 120,
        quality_level: float = 0.01,
        min_distance: int = 8,
        min_inliers: int = 8,
        max_shift_ratio: float = 0.4,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_inliers = min_inliers
        self.max_shift_ratio = max_shift_ratio
        self._previous: np.ndarray | None = None

    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        current = frame.normalized
        if self._previous is None:
            self._previous = current.copy()
            return GlobalMotion()

        previous_points = cv2.goodFeaturesToTrack(
            self._previous,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if previous_points is None or len(previous_points) < self.min_inliers:
            self._previous = current.copy()
            return GlobalMotion(valid=False)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(self._previous, current, previous_points, None)
        if next_points is None or status is None:
            self._previous = current.copy()
            return GlobalMotion(valid=False)

        good_previous = previous_points[status.ravel() == 1]
        good_next = next_points[status.ravel() == 1]
        if len(good_previous) < self.min_inliers:
            self._previous = current.copy()
            return GlobalMotion(valid=False)

        matrix, inliers = cv2.estimateAffinePartial2D(good_previous, good_next)
        self._previous = current.copy()
        if matrix is None or inliers is None:
            return GlobalMotion(valid=False)

        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        inlier_ratio = float(np.count_nonzero(inliers)) / max(len(inliers), 1)
        max_shift = max(frame.bgr.shape[:2]) * self.max_shift_ratio
        valid = abs(dx) <= max_shift and abs(dy) <= max_shift and inlier_ratio > 0.35
        return GlobalMotion(dx=dx, dy=dy, response=inlier_ratio, valid=valid)
