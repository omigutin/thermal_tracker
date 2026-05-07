"""
    Фильтр кандидатов по локальному контрасту.
    Модуль содержит атомарный фильтр, который сравнивает яркость объекта с яркостью локального фона вокруг bbox.
    Фильтр полезен для тепловизионных и монохромных кадров, где реальная цель
    обычно должна отличаться от ближайшего окружения по интенсивности.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .base_candidate_filter import BaseCandidateFilter


@dataclass
class ContrastCandidateFilter(BaseCandidateFilter):
    """Фильтр кандидатов по контрасту объекта относительно локального фона."""

    min_contrast: float = 6.0
    border: int = 6

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        """ Удаляет объекты с недостаточным контрастом относительно фона. """

        normalized = frame.normalized.astype(np.float32)
        result: list[DetectedObject] = []
        for obj in objects:
            bbox = obj.bbox.clamp(normalized.shape)
            object_patch = normalized[bbox.y:bbox.y2, bbox.x:bbox.x2]
            if object_patch.size == 0:
                continue

            ring_bbox = bbox.pad(self.border, self.border).clamp(normalized.shape)
            ring_patch = normalized[ring_bbox.y:ring_bbox.y2, ring_bbox.x:ring_bbox.x2]
            ring_mask = np.ones(ring_patch.shape, dtype=bool)
            y1 = bbox.y - ring_bbox.y
            y2 = y1 + bbox.height
            x1 = bbox.x - ring_bbox.x
            x2 = x1 + bbox.width
            ring_mask[y1:y2, x1:x2] = False
            background = ring_patch[ring_mask]
            if background.size < 10:
                result.append(obj)
                continue

            contrast = abs(float(np.median(object_patch)) - float(np.median(background)))
            if contrast >= self.min_contrast:
                result.append(obj)
        return result
