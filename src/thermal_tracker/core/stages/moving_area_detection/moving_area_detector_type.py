"""Типы методов обнаружения движущихся областей."""

from __future__ import annotations

from enum import StrEnum


class MovingAreaDetectorType(StrEnum):
    """Доступные методы стадии moving_area_detection."""

    OPENCV_MOG2 = "opencv_mog2"  # Background subtraction MOG2 для поиска изменений относительно фона.
    OPENCV_KNN = "opencv_knn"  # Background subtraction KNN, иногда устойчивее на сложном фоне.
    OPENCV_FRAME_DIFFERENCE = "opencv_frame_difference"  # Ищет движение по разности соседних кадров.
    OPENCV_RUNNING_AVERAGE = "opencv_running_average"  # Сравнивает кадр с медленно обновляемой моделью фона.

    # OPENCV_OPTICAL_FLOW = "opencv_optical_flow"  # Поиск движения по полю оптического потока.
    # THERMAL_CHANGE = "thermal_change"  # Детекция изменений с учетом теплового контраста сцены.
