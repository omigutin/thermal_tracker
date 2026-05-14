from __future__ import annotations

import cv2
import numpy as np

from .....domain.models import BoundingBox


class ImagePatch:
    """Работает с локальными участками изображения для template tracking."""

    @staticmethod
    def safe_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Изменить размер изображения без нулевой ширины или высоты."""
        width, height = size
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
        """Вырезать bbox из изображения или вернуть None для вырожденного участка."""
        clamped = bbox.clamp(image.shape)

        if clamped.width <= 1 or clamped.height <= 1:
            return None

        return image[clamped.y:clamped.y2, clamped.x:clamped.x2]

    @staticmethod
    def correlation(image: np.ndarray, template: np.ndarray) -> float:
        """Посчитать нормализованную корреляцию двух одинаковых patch."""
        if image.shape != template.shape:
            return 0.0

        if float(np.std(image)) < 1e-6 or float(np.std(template)) < 1e-6:
            return 0.0

        score_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        return float(score_map[0, 0])
