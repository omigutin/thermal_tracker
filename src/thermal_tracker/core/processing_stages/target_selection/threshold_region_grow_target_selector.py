"""Инициализация цели через рост области от клика."""

from __future__ import annotations

from .opencv_click_target_selector import ClickTargetSelector


class ThresholdRegionGrowClickInitializer(ClickTargetSelector):
    """Пока переиспользует текущую рабочую гибридную реализацию."""
