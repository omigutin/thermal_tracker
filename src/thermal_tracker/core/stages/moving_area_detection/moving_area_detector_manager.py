"""Менеджер выбора детектора движущихся областей."""

from __future__ import annotations

from ...domain.models import GlobalMotion, MotionDetectionResult, ProcessedFrame
from .base_moving_area_detector import BaseMotionDetector
from .moving_area_detector_type import MovingAreaDetectorType
from .opencv_frame_difference_detector import FrameDifferenceMotionDetector
from .opencv_knn_detector import KnnMotionDetector
from .opencv_mog2_detector import Mog2MotionDetector
from .opencv_running_average_detector import RunningAverageMotionDetector


MovingAreaDetectorInput = MovingAreaDetectorType | str


class MovingAreaDetectorManager:
    """Создаёт и запускает выбранный детектор движущихся областей."""

    def __init__(self, detector: MovingAreaDetectorInput) -> None:
        self._detector_type = self._normalize_detector_type(detector)
        self._detector = self._build_detector(self._detector_type)

    @property
    def detector(self) -> BaseMotionDetector:
        """Возвращает подготовленный детектор."""

        return self._detector

    def detect(self, frame: ProcessedFrame, motion: GlobalMotion) -> MotionDetectionResult:
        """Возвращает маску или результат обнаружения движения."""

        result = self._detector.detect(frame, motion)
        result.source_name = self._detector_type.value
        return result

    @classmethod
    def _build_detector(cls, detector_type: MovingAreaDetectorType) -> BaseMotionDetector:
        if detector_type == MovingAreaDetectorType.OPENCV_MOG2:
            return Mog2MotionDetector()
        if detector_type == MovingAreaDetectorType.OPENCV_KNN:
            return KnnMotionDetector()
        if detector_type == MovingAreaDetectorType.OPENCV_FRAME_DIFFERENCE:
            return FrameDifferenceMotionDetector()
        if detector_type == MovingAreaDetectorType.OPENCV_RUNNING_AVERAGE:
            return RunningAverageMotionDetector()
        raise ValueError(f"Unsupported moving area detector type: {detector_type!r}.")

    @staticmethod
    def _normalize_detector_type(detector: MovingAreaDetectorInput) -> MovingAreaDetectorType:
        if isinstance(detector, MovingAreaDetectorType):
            return detector
        try:
            return MovingAreaDetectorType(detector)
        except ValueError:
            pass
        detector_by_name = MovingAreaDetectorType.__members__.get(detector.upper())
        if detector_by_name is not None:
            return detector_by_name
        raise ValueError(
            f"Unsupported moving area detector value: {detector!r}. "
            f"Available values: {tuple(item.value for item in MovingAreaDetectorType)}."
        )
