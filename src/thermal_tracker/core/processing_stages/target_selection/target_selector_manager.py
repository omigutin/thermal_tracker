"""Менеджер выбора метода инициализации цели."""

from __future__ import annotations

from ...config import ClickSelectionConfig
from ...domain.models import BoundingBox, ProcessedFrame, SelectionResult
from .base_target_selector import BaseClickInitializer
from .connected_component_target_selector import ConnectedComponentClickInitializer
from .edge_guided_target_selector import EdgeGuidedClickInitializer
from .opencv_click_target_selector import ClickTargetSelector
from .target_selector_type import TargetSelectorType
from .threshold_region_grow_target_selector import ThresholdRegionGrowClickInitializer


TargetSelectorInput = TargetSelectorType | str


class TargetSelectorManager:
    """Создаёт и запускает выбранный метод выбора цели по клику."""

    def __init__(self, selector: TargetSelectorInput, config: ClickSelectionConfig) -> None:
        self._selector = self._build_selector(selector, config)

    @property
    def selector(self) -> BaseClickInitializer:
        """Возвращает подготовленный selector."""

        return self._selector

    def select(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> SelectionResult:
        """Находит цель вокруг точки клика."""

        return self._selector.select(frame, point, expected_bbox)

    @classmethod
    def _build_selector(
        cls,
        selector: TargetSelectorInput,
        config: ClickSelectionConfig,
    ) -> BaseClickInitializer:
        selector_type = cls._normalize_selector_type(selector)
        if selector_type == TargetSelectorType.OPENCV_CLICK:
            return ClickTargetSelector(config)
        if selector_type == TargetSelectorType.CONNECTED_COMPONENT:
            return ConnectedComponentClickInitializer(config)
        if selector_type == TargetSelectorType.EDGE_GUIDED:
            return EdgeGuidedClickInitializer(config)
        if selector_type == TargetSelectorType.THRESHOLD_REGION_GROW:
            return ThresholdRegionGrowClickInitializer(config)
        raise ValueError(f"Unsupported target selector type: {selector_type!r}.")

    @staticmethod
    def _normalize_selector_type(selector: TargetSelectorInput) -> TargetSelectorType:
        if isinstance(selector, TargetSelectorType):
            return selector
        try:
            return TargetSelectorType(selector)
        except ValueError:
            pass
        selector_by_name = TargetSelectorType.__members__.get(selector.upper())
        if selector_by_name is not None:
            return selector_by_name
        raise ValueError(
            f"Unsupported target selector value: {selector!r}. "
            f"Available values: {tuple(item.value for item in TargetSelectorType)}."
        )
