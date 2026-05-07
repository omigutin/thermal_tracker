"""Инициализация с упором на границы объекта."""

from __future__ import annotations

from .opencv_click_target_selector import ClickTargetSelector


class EdgeGuidedClickInitializer(ClickTargetSelector):
    """Пока использует текущую рабочую гибридную базу."""

    implementation_name = "edge_guided"
    is_ready = True
