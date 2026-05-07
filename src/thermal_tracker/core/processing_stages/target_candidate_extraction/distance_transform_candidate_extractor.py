"""Заготовка под split на основе distance transform."""

from __future__ import annotations

from .base_candidate_extractor import BaseObjectBuilder


class DistanceTransformSplitObjectBuilder(BaseObjectBuilder):
    """Будущий builder, который умеет делить плотные скопления."""

    def build(self, frame, detection):
        raise NotImplementedError("Distance-transform splitter пока не реализован.")
