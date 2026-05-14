from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self, Optional

import cv2
import numpy as np

from ....config.preset_field_reader import PresetFieldReader
from ....domain.models import ProcessedFrame
from ..result import FrameStabilizerResult
from ..type import FrameStabilizerType
from .base_frame_stabilizer import BaseFrameStabilizer


@dataclass(frozen=True, slots=True)
class FeatureAffineFrameStabilizerConfig:
    """Хранит настройки стабилизации кадра по опорным точкам и аффинному преобразованию."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[FrameStabilizerType] = FrameStabilizerType.FEATURE_AFFINE
    # Максимальное количество опорных точек для поиска.
    max_corners: int = 120
    # Минимальное качество точки для cv2.goodFeaturesToTrack.
    quality_level: float = 0.01
    # Минимальная дистанция между найденными точками.
    min_distance: int = 8
    # Минимальное количество хороших совпадений для оценки движения.
    min_inliers: int = 8
    # Максимально допустимый сдвиг относительно размера кадра.
    max_shift_ratio: float = 0.35
    # Минимальная доля inlier-точек для признания результата корректным.
    min_inlier_ratio: float = 0.35

    def __post_init__(self) -> None:
        """Проверить корректность параметров стабилизации кадра."""
        if self.max_corners <= 0:
            raise ValueError("max_corners must be greater than 0.")
        if not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be in range (0, 1].")
        if self.min_distance <= 0:
            raise ValueError("min_distance must be greater than 0.")
        if self.min_inliers <= 0:
            raise ValueError("min_inliers must be greater than 0.")
        if self.min_inliers > self.max_corners:
            raise ValueError("min_inliers must be less than or equal to max_corners.")
        if not 0 < self.max_shift_ratio <= 1:
            raise ValueError("max_shift_ratio must be in range (0, 1].")
        if not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be in range (0, 1].")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "max_corners")
        reader.pop_float_to(kwargs, "quality_level")
        reader.pop_int_to(kwargs, "min_distance")
        reader.pop_int_to(kwargs, "min_inliers")
        reader.pop_float_to(kwargs, "max_shift_ratio")
        reader.pop_float_to(kwargs, "min_inlier_ratio")
        reader.ensure_empty()

        return cls(**kwargs)


@dataclass(slots=True)
class FeatureAffineFrameStabilizer(BaseFrameStabilizer):
    """Оценивает движение камеры по опорным точкам и аффинному преобразованию."""

    config: FeatureAffineFrameStabilizerConfig
    _previous: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def apply(self, frame: ProcessedFrame) -> FrameStabilizerResult:
        """Вернуть оценку смещения текущего кадра относительно предыдущего."""
        current = frame.normalized

        if self._previous is None:
            self._previous = current.copy()
            return FrameStabilizerResult()

        previous_points = cv2.goodFeaturesToTrack(
            self._previous,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
        )

        if previous_points is None or len(previous_points) < self.config.min_inliers:
            self._previous = current.copy()
            return FrameStabilizerResult(valid=False)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous,
            current,
            previous_points,
            None,
        )

        if next_points is None or status is None:
            self._previous = current.copy()
            return FrameStabilizerResult(valid=False)

        good_previous = previous_points[status.ravel() == 1]
        good_next = next_points[status.ravel() == 1]

        if len(good_previous) < self.config.min_inliers:
            self._previous = current.copy()
            return FrameStabilizerResult(valid=False)

        matrix, inliers = cv2.estimateAffinePartial2D(good_previous, good_next)
        self._previous = current.copy()

        if matrix is None or inliers is None:
            return FrameStabilizerResult(valid=False)

        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        inlier_ratio = float(np.count_nonzero(inliers)) / max(len(inliers), 1)
        max_shift = max(frame.bgr.shape[:2]) * self.config.max_shift_ratio

        valid = (
            abs(dx) <= max_shift
            and abs(dy) <= max_shift
            and inlier_ratio >= self.config.min_inlier_ratio
        )

        return FrameStabilizerResult(dx=dx, dy=dy, response=inlier_ratio, valid=valid)
