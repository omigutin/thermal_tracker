"""Заготовка под blob-based сборку объектов."""

from __future__ import annotations

from .base_candidate_extractor import BaseObjectBuilder


class BlobObjectBuilder(BaseObjectBuilder):
    """Будущий builder для blob-детектора и похожих компактных объектов."""

    implementation_name = "blob"
    is_ready = False

    def build(self, frame, detection):
        raise NotImplementedError("Blob-based object builder пока не реализован.")
