"""Заготовка под фильтр по живучести объекта во времени."""

from __future__ import annotations

from .base_candidate_filter import BaseTargetFilter


class PersistenceTargetFilter(BaseTargetFilter):
    """Будущий фильтр, который будет выкидывать одноразовые вспышки."""

    implementation_name = "persistence"
    is_ready = False

    def filter(self, frame, objects, motion):
        raise NotImplementedError("Persistence filter пока не реализован.")
