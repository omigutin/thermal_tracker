"""Инициализация цели через connected components."""

from __future__ import annotations

from .opencv_click_target_selector import ClickTargetSelector


class ConnectedComponentClickInitializer(ClickTargetSelector):
    """Пока переиспользует текущую рабочую гибридную реализацию."""

    implementation_name = "connected_component"
    is_ready = True
