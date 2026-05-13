"""Инициализация цели через connected components."""

from __future__ import annotations

from .opencv_click_target_selector import ClickTargetSelector


class ConnectedComponentClickInitializer(ClickTargetSelector):
    """Пока переиспользует текущую рабочую гибридную реализацию."""
