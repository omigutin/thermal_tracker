"""Типы single-target трекеров."""

from __future__ import annotations

from enum import StrEnum


class TargetTrackerType(StrEnum):
    """Доступные методы стадии target_tracking."""

    OPENCV_TEMPLATE_POINT = "opencv_template_point"  # Гибрид шаблонного поиска, опорных точек и модели движения.
    NN_YOLO = "nn_yolo"  # Сопровождение выбранной цели через YOLO-детекции и внешний NN-интерфейс.
    IRST_CONTRAST = "irst_contrast"  # IRST: локальный контраст + фильтр Калмана, без шаблонов и optical flow.

    # OPENCV_KCF = "opencv_kcf"  # Быстрый классический OpenCV-трекер для одной цели.
    # OPENCV_CSRT = "opencv_csrt"  # Более точный, но более тяжелый OpenCV-трекер.
    # OPENCV_MOSSE = "opencv_mosse"  # Очень быстрый correlation filter tracker.
    # OPENCV_MEDIAN_FLOW = "opencv_median_flow"  # Трекер для плавного движения без резких пропаданий.
    # OPENCV_TEMPLATE = "opencv_template"  # Чистый шаблонный поиск без опорных точек.
    # POINT_FLOW = "point_flow"  # Сопровождение по оптическому потоку опорных точек.
