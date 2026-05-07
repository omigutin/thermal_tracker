"""Инициализация цели по клику."""

from .base_target_selector import BaseClickInitializer
from .opencv_click_target_selector import ClickTargetSelector
from .connected_component_target_selector import ConnectedComponentClickInitializer
from .edge_guided_target_selector import EdgeGuidedClickInitializer
from .threshold_region_grow_target_selector import ThresholdRegionGrowClickInitializer
from .watershed_seed_target_selector import WatershedSeedClickInitializer

__all__ = [
    "BaseClickInitializer",
    "ClickTargetSelector",
    "ConnectedComponentClickInitializer",
    "EdgeGuidedClickInitializer",
    "ThresholdRegionGrowClickInitializer",
    "WatershedSeedClickInitializer",
]
