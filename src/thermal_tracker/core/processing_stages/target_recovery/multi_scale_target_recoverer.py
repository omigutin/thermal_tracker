"""Заготовка под multiscale-повторный захват."""

from __future__ import annotations

from .base_target_recoverer import BaseReacquirer


class MultiScaleReacquirer(BaseReacquirer):
    """Будущий reacquirer с перебором нескольких масштабов и областей поиска."""

    implementation_name = "multi_scale"
    is_ready = False

    def reacquire(self, frame, last_bbox, motion):
        raise NotImplementedError("Multi-scale reacquirer пока не реализован.")
