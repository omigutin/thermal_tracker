from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .....domain.models import BoundingBox, ProcessedFrame
from .point_prediction import PointPrediction
from .template_point_config import TemplatePointTargetTrackerConfig


@dataclass(slots=True)
class FeaturePointTracker:
    """Ведёт опорные точки внутри цели и строит прогноз bbox по их движению."""

    config: TemplatePointTargetTrackerConfig
    previous_normalized: np.ndarray | None = field(default=None, init=False, repr=False)
    tracked_points: np.ndarray | None = field(default=None, init=False, repr=False)
    point_center_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32),
        init=False,
        repr=False,
    )
    frames_since_refresh: int = field(default=0, init=False)

    def reset(self) -> None:
        """Сбросить состояние сопровождения по точкам."""
        self.previous_normalized = None
        self.tracked_points = None
        self.point_center_offset[:] = 0.0
        self.frames_since_refresh = 0

    def set_previous_frame(self, frame: ProcessedFrame) -> None:
        """Запомнить текущий кадр как предыдущий для следующего optical flow."""
        self.previous_normalized = frame.normalized.copy()

    def initialize(self, frame: ProcessedFrame, bbox: BoundingBox, *, force: bool = False) -> None:
        """Набрать хорошие точки внутри bbox цели."""
        if (
            not force
            and self.tracked_points is not None
            and len(self.tracked_points) >= self.config.min_feature_points
        ):
            return

        mask = np.zeros_like(frame.normalized, dtype=np.uint8)
        clamped = bbox.clamp(frame.bgr.shape)
        mask[clamped.y:clamped.y2, clamped.x:clamped.x2] = 255

        points = cv2.goodFeaturesToTrack(
            frame.gradient,
            maxCorners=self.config.max_feature_points,
            qualityLevel=self.config.feature_quality_level,
            minDistance=self.config.feature_min_distance,
            mask=mask,
            blockSize=3,
        )

        if points is None:
            points = cv2.goodFeaturesToTrack(
                frame.normalized,
                maxCorners=self.config.max_feature_points,
                qualityLevel=self.config.feature_quality_level,
                minDistance=self.config.feature_min_distance,
                mask=mask,
                blockSize=3,
            )

        if points is None or len(points) < self.config.min_feature_points:
            self.tracked_points = None
            return

        self.tracked_points = points.astype(np.float32)
        self._update_point_offset(clamped)
        self.frames_since_refresh = 0

    def predict(self, frame: ProcessedFrame, bbox: BoundingBox) -> PointPrediction | None:
        """Спрогнозировать bbox цели по опорным точкам."""
        if self.previous_normalized is None or self.tracked_points is None:
            return None

        if len(self.tracked_points) < self.config.min_feature_points:
            self.tracked_points = None
            return None

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.previous_normalized,
            frame.normalized,
            self.tracked_points,
            None,
            winSize=(
                self.config.optical_flow_window_size,
                self.config.optical_flow_window_size,
            ),
            maxLevel=self.config.optical_flow_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.optical_flow_criteria_count,
                self.config.optical_flow_criteria_eps,
            ),
        )

        if next_points is None or status is None:
            self.tracked_points = None
            return None

        valid_mask = status.flatten() == 1

        if errors is not None:
            valid_mask &= errors.flatten() < self.config.optical_flow_max_error

        previous_points = self.tracked_points.reshape(-1, 2)[valid_mask]
        current_points = next_points.reshape(-1, 2)[valid_mask]

        if len(current_points) < self.config.min_feature_points:
            self.tracked_points = None
            return None

        current_points = self._filter_consistent_points(
            previous_points=previous_points,
            current_points=current_points,
        )

        if len(current_points) < self.config.min_feature_points:
            self.tracked_points = None
            return None

        self.tracked_points = current_points.reshape(-1, 1, 2).astype(np.float32)

        point_center = np.median(current_points, axis=0) + self.point_center_offset
        predicted_bbox = BoundingBox.from_center(
            point_center[0],
            point_center[1],
            bbox.width,
            bbox.height,
        ).clamp(frame.bgr.shape)

        confidence = min(
            1.0,
            len(current_points) / max(float(self.config.max_feature_points), 1.0),
        )

        return PointPrediction(bbox=predicted_bbox, confidence=confidence)

    def refresh(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Освежить набор точек, если старые точки закончились или устарели."""
        usable_points = self._filter_points_inside_bbox(bbox)

        if usable_points is not None:
            self.tracked_points = usable_points
            self._update_point_offset(bbox)

        self.frames_since_refresh += 1

        should_refresh = (
            self.tracked_points is None
            or len(self.tracked_points) < self.config.min_feature_points
            or self.frames_since_refresh >= self.config.feature_refresh_interval
        )

        if should_refresh:
            self.initialize(frame=frame, bbox=bbox, force=True)

    def _filter_consistent_points(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
    ) -> np.ndarray:
        """Оставить точки с согласованным смещением."""
        displacements = current_points - previous_points
        median_displacement = np.median(displacements, axis=0)
        residuals = np.linalg.norm(displacements - median_displacement, axis=1)

        allowed_residual = max(3.0, float(np.median(residuals) * 2.5 + 1.0))
        consistent_mask = residuals <= allowed_residual

        return current_points[consistent_mask]

    def _filter_points_inside_bbox(self, bbox: BoundingBox) -> np.ndarray | None:
        """Оставить только точки, которые всё ещё лежат рядом с текущим bbox."""
        if self.tracked_points is None:
            return None

        x1 = bbox.x - 3
        y1 = bbox.y - 3
        x2 = bbox.x2 + 3
        y2 = bbox.y2 + 3

        points = self.tracked_points.reshape(-1, 2)
        inside_mask = (
            (points[:, 0] >= x1)
            & (points[:, 0] <= x2)
            & (points[:, 1] >= y1)
            & (points[:, 1] <= y2)
        )
        filtered = points[inside_mask]

        if len(filtered) < self.config.min_feature_points:
            return None

        return filtered.reshape(-1, 1, 2).astype(np.float32)

    def _update_point_offset(self, bbox: BoundingBox) -> None:
        """Запомнить смещение центра bbox относительно центра набора точек."""
        if self.tracked_points is None or len(self.tracked_points) == 0:
            self.point_center_offset[:] = 0.0
            return

        point_center = np.median(self.tracked_points.reshape(-1, 2), axis=0)
        bbox_center = np.array(bbox.center, dtype=np.float32)
        self.point_center_offset = bbox_center - point_center
        