"""
    Фильтр кандидатов по площади и пропорциям bbox.
    Модуль содержит атомарный фильтр, который отбрасывает объекты-кандидаты по простым геометрическим признакам:
        - слишком маленькая площадь;
        - слишком большая площадь относительно кадра;
        - слишком вытянутая форма bbox.
    Фильтр полезен как ранний дешёвый этап очистки ложных срабатываний.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .base_candidate_filter import BaseCandidateFilter


@dataclass
class AreaAspectCandidateFilter(BaseCandidateFilter):
    """ Фильтр кандидатов по площади и соотношению сторон bbox. """

    min_area: int = 24
    max_frame_fraction: float = 0.25
    max_aspect_ratio: float = 6.0

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        """ Удаляет объекты с недопустимой площадью или пропорциями bbox. """

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