"""Обнаружение движения."""

from .base_moving_area_detector import BaseMotionDetector
from .opencv_frame_difference_detector import FrameDifferenceMotionDetector
from .opencv_knn_detector import KnnMotionDetector
from .opencv_mog2_detector import Mog2MotionDetector
from .optical_flow_detector import OpticalFlowMotionDetector
from .opencv_running_average_detector import RunningAverageMotionDetector
from .thermal_change_detector import ThermalChangeMotionDetector

__all__ = [
    "BaseMotionDetector",
    "FrameDifferenceMotionDetector",
    "KnnMotionDetector",
    "Mog2MotionDetector",
    "OpticalFlowMotionDetector",
    "RunningAverageMotionDetector",
    "ThermalChangeMotionDetector",
]
