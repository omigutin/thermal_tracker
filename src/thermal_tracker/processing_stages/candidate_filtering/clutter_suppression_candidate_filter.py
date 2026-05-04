"""Заготовка под подавление clutter-сцены."""

from __future__ import annotations

from .base_candidate_filter import BaseTargetFilter


class ClutterSuppressionTargetFilter(BaseTargetFilter):
    """Будущий фильтр для плотных сцен с множеством похожих горячих структур."""

    implementation_name = "clutter_suppression"
    is_ready = False

    def filter(self, frame, objects, motion):
        raise NotImplementedError("Clutter-suppression filter пока не реализован.")
