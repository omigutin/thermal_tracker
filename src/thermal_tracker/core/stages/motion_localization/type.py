from __future__ import annotations

from enum import StrEnum


class MotionLocalizationType(StrEnum):
    """Типы операций локализации движения."""
    FRAME_DIFFERENCE = "frame_difference"  # Ищет движение по разности соседних кадров.
    KNN = "knn"  # Background subtraction KNN, иногда устойчивее на сложном фоне.
    MOG2 = "mog2"  # Background subtraction MOG2 для поиска изменений относительно фона.
    RUNNING_AVERAGE = "running_average"  # Сравнивает кадр с медленно обновляемой моделью фона.
