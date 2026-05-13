"""
    Фильтр кандидатов по касанию границ кадра.
    Модуль содержит атомарный фильтр, который отбрасывает объекты-кандидаты,
    если их bbox находится слишком близко к границе кадра.
    Такие объекты часто являются неполными, обрезанными или нестабильными ложными срабатываниями.
"""

from __future__ import annotations

from dataclasses import dataclass

from ....domain.models import DetectedObject, GlobalMotion, ProcessedFrame
from .base_candidate_filter import BaseCandidateFilter


@dataclass
class BorderTouchCandidateFilter(BaseCandidateFilter):
    """ Фильтр кандидатов, касающихся границы кадра. """

    border_margin: int = 2

    def filter(self, frame: ProcessedFrame, objects: list[DetectedObject], motion: GlobalMotion) -> list[DetectedObject]:
        """ Удаляет объекты, расположенные слишком близко к границе кадра. """

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
