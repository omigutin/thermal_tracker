"""Сборка объектов через connected components."""

from __future__ import annotations

import cv2

from ...domain.models import BoundingBox, DetectedObject, MotionDetectionResult, ProcessedFrame
from .base_candidate_extractor import BaseObjectBuilder


class ConnectedComponentsObjectBuilder(BaseObjectBuilder):
    """Строит объекты напрямую из компонент связности."""

    def __init__(self, min_area: int = 24) -> None:
        self.min_area = min_area

    def build(self, frame: ProcessedFrame, detection: MotionDetectionResult) -> list[DetectedObject]:
        _, _, stats, _ = cv2.connectedComponentsWithStats(detection.mask)
        objects: list[DetectedObject] = []
        for label in range(1, stats.shape[0]):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            bbox = BoundingBox(
                x=int(stats[label, cv2.CC_STAT_LEFT]),
                y=int(stats[label, cv2.CC_STAT_TOP]),
                width=int(stats[label, cv2.CC_STAT_WIDTH]),
                height=int(stats[label, cv2.CC_STAT_HEIGHT]),
            ).clamp(frame.bgr.shape)
            objects.append(
                DetectedObject(
                    bbox=bbox,
                    area=area,
                    confidence=min(1.0, area / max(self.min_area, 1)),
                    label="motion_component",
                )
            )
        return objects
