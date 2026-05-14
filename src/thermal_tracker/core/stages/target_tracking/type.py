from __future__ import annotations

from enum import StrEnum


class TargetTrackerType(StrEnum):
    """Типы single-target трекеров."""
    # Сопровождает цель по шаблонам, опорным точкам и прогнозу движения.
    TEMPLATE_POINT = "template_point"
    # Сопровождает цель через OpenCV CSRT-трекер.
    CSRT = "csrt"
    # Сопровождает выбранную цель через YOLO-детекции и внешний NN-интерфейс.
    YOLO = "yolo"
    # Сопровождает маленькую тепловую цель через локальный контраст и фильтр Калмана.
    IRST_CONTRAST = "irst_contrast"
