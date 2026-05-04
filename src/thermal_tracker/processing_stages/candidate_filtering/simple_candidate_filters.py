"""Набор простых, но полезных фильтров ложных целей."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .base_candidate_filter import BaseTargetFilter


class FilterChain(BaseTargetFilter):
    """Прогоняет объекты через несколько фильтров подряд."""

    implementation_name = "filter_chain"
    is_ready = True

    def __init__(self, filters: list[BaseTargetFilter]) -> None:
        self.filters = filters

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        current = list(objects)
        for filter_stage in self.filters:
            current = filter_stage.filter(frame, current, motion)
        return current


@dataclass
class AreaAspectTargetFilter(BaseTargetFilter):
    """Убирает слишком мелкие, слишком большие и слишком вытянутые объекты."""

    min_area: int = 24
    max_frame_fraction: float = 0.25
    max_aspect_ratio: float = 6.0

    implementation_name = "area_aspect"
    is_ready = True

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        frame_area = frame.bgr.shape[0] * frame.bgr.shape[1]
        result: list[DetectedObject] = []
        for obj in objects:
            aspect = max(obj.bbox.width, obj.bbox.height) / max(min(obj.bbox.width, obj.bbox.height), 1)
            if obj.area < self.min_area:
                continue
            if obj.area > frame_area * self.max_frame_fraction:
                continue
            if aspect > self.max_aspect_ratio:
                continue
            result.append(obj)
        return result


@dataclass
class BorderTouchTargetFilter(BaseTargetFilter):
    """Убирает объекты, упёршиеся в край кадра."""

    border_margin: int = 2

    implementation_name = "border_touch"
    is_ready = True

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        frame_h, frame_w = frame.bgr.shape[:2]
        result: list[DetectedObject] = []
        for obj in objects:
            if obj.bbox.x <= self.border_margin:
                continue
            if obj.bbox.y <= self.border_margin:
                continue
            if obj.bbox.x2 >= frame_w - self.border_margin:
                continue
            if obj.bbox.y2 >= frame_h - self.border_margin:
                continue
            result.append(obj)
        return result


@dataclass
class ContrastTargetFilter(BaseTargetFilter):
    """Проверяет, отличается ли объект от локального фона по яркости."""

    min_contrast: float = 6.0
    border: int = 6

    implementation_name = "contrast"
    is_ready = True

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
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
