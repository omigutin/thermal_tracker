"""Заготовка под почти глобальный повторный захват."""

from __future__ import annotations

from .base_target_recoverer import BaseReacquirer


class GlobalReacquirer(BaseReacquirer):
    """Будущий fallback-режим, когда локальный поиск уже не помог."""

    implementation_name = "global"
    is_ready = False

    def reacquire(self, frame, last_bbox, motion):
        raise NotImplementedError("Global reacquirer пока не реализован.")
