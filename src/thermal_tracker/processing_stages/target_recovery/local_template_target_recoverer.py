"""Заготовка под локальный повторный захват по шаблону."""

from __future__ import annotations

from .base_target_recoverer import BaseReacquirer


class LocalTemplateReacquirer(BaseReacquirer):
    """Отдельная стадия для локального reacquire.

    Сейчас похожая логика уже живёт внутри гибридного single-target tracker.
    Отдельным классом мы выносим её архитектурно, чтобы позже можно было
    переключать стратегии независимо от основного трекера.
    """

    implementation_name = "local_template"
    is_ready = False

    def reacquire(self, frame, last_bbox, motion):
        raise NotImplementedError("Отдельный local reacquirer пока не вынесен из текущего трекера.")
