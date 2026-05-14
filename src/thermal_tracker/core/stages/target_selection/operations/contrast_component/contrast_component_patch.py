from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....domain.models import BoundingBox


@dataclass(slots=True)
class ContrastComponentPatch:
    """Локальный участок кадра вокруг точки выбора цели."""
    image: np.ndarray
    origin_x: int
    origin_y: int
    local_x: int
    local_y: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Вернуть форму локального участка."""
        return self.image.shape

    @property
    def height(self) -> int:
        """Вернуть высоту локального участка."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Вернуть ширину локального участка."""
        return int(self.image.shape[1])

    @property
    def global_x(self) -> int:
        """Вернуть глобальную координату X точки выбора."""
        return self.origin_x + self.local_x

    @property
    def global_y(self) -> int:
        """Вернуть глобальную координату Y точки выбора."""
        return self.origin_y + self.local_y

    def to_global_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """Перевести локальный bbox участка в координаты полного кадра."""
        return BoundingBox(
            x=self.origin_x + bbox.x,
            y=self.origin_y + bbox.y,
            width=bbox.width,
            height=bbox.height,
        )

    def to_local_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """Перевести bbox полного кадра в локальные координаты участка."""
        return BoundingBox(
            x=bbox.x - self.origin_x,
            y=bbox.y - self.origin_y,
            width=bbox.width,
            height=bbox.height,
        )
