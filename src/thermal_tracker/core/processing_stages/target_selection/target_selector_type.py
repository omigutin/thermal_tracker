"""Типы методов выбора цели по клику."""

from __future__ import annotations

from enum import StrEnum


class TargetSelectorType(StrEnum):
    """Доступные методы стадии target_selection."""

    OPENCV_CLICK = "opencv_click"  # Гибридный выбор цели вокруг клика по контрасту и компонентам.
    CONNECTED_COMPONENT = "connected_component"  # Выбирает компоненту связности, связанную с точкой клика.
    EDGE_GUIDED = "edge_guided"  # Уточняет выбор цели по локальным границам объекта.
    THRESHOLD_REGION_GROW = "threshold_region_grow"  # Растит область от клика по порогу похожести яркости.

    # WATERSHED_SEED = "watershed_seed"  # Использует клик как seed для водораздела.
