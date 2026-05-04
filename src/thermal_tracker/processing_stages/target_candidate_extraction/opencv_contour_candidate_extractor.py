"""Сборка объектов через контуры."""

from __future__ import annotations

import cv2

from ...domain.models import BoundingBox, DetectedObject, MotionDetectionResult, ProcessedFrame
from .base_candidate_extractor import BaseObjectBuilder


class ContourObjectBuilder(BaseObjectBuilder):
    """Ищет внешние контуры и упаковывает их в объекты."""

    implementation_name = "contour"
    is_ready = True

    def __init__(self, min_area: int = 24) -> None:
        self.min_area = min_area

    def build(self, frame: ProcessedFrame, detection: MotionDetectionResult) -> list[DetectedObject]:
        contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects: list[DetectedObject] = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(x=x, y=y, width=w, height=h).clamp(frame.bgr.shape)
            objects.append(
                DetectedObject(
                    bbox=bbox,
                    area=area,
                    confidence=min(1.0, area / max(self.min_area, 1)),
                    label="motion_contour",
                    source=self.implementation_name,
                )
            )
        return objects
