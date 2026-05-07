"""Заготовка под watershed-инициализацию от seed-точки."""

from __future__ import annotations

from .base_target_selector import BaseClickInitializer


class WatershedSeedClickInitializer(BaseClickInitializer):
    """Будущая реализация для более агрессивного разделения соседних целей."""

    implementation_name = "watershed_seed"
    is_ready = False

    def select(self, frame, point, expected_bbox=None):
        raise NotImplementedError("Watershed-инициализация пока не реализована.")
