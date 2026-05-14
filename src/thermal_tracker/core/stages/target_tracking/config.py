from __future__ import annotations

from typing import TypeAlias

from .operations import (
    CsrtTargetTrackerConfig,
    IrstContrastTargetTrackerConfig,
    TemplatePointTargetTrackerConfig,
    YoloTargetTrackerConfig,
)


TargetTrackerConfig: TypeAlias = (
    TemplatePointTargetTrackerConfig
    | CsrtTargetTrackerConfig
    | IrstContrastTargetTrackerConfig
    | YoloTargetTrackerConfig
)


_TargetTrackerConfigClass: TypeAlias = (
    type[TemplatePointTargetTrackerConfig]
    | type[CsrtTargetTrackerConfig]
    | type[IrstContrastTargetTrackerConfig]
    | type[YoloTargetTrackerConfig]
)


TARGET_TRACKER_CONFIG_CLASSES: dict[str, _TargetTrackerConfigClass] = {
    str(TemplatePointTargetTrackerConfig.operation_type): TemplatePointTargetTrackerConfig,
    str(CsrtTargetTrackerConfig.operation_type): CsrtTargetTrackerConfig,
    str(IrstContrastTargetTrackerConfig.operation_type): IrstContrastTargetTrackerConfig,
    str(YoloTargetTrackerConfig.operation_type): YoloTargetTrackerConfig,
}
