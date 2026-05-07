"""Заготовка под watershed-сборку объектов."""

from __future__ import annotations

from .base_candidate_extractor import BaseObjectBuilder


class WatershedObjectBuilder(BaseObjectBuilder):
    """Будущий builder для разделения слипшихся целей."""

    implementation_name = "watershed"
    is_ready = False

    def build(self, frame, detection):
        raise NotImplementedError("Watershed object builder пока не реализован.")
