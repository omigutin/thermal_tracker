from __future__ import annotations

from .base_target_tracker import BaseTargetTracker
from .csrt_target_tracker import (
    CsrtTargetTracker,
    CsrtTargetTrackerConfig,
)
from .irst_contrast_target_tracker import (
    IrstContrastTargetTracker,
    IrstContrastTargetTrackerConfig,
)
from .template_point_target_tracker import (
    TemplatePointTargetTracker,
    TemplatePointTargetTrackerConfig,
)
from .yolo_target_tracker import (
    YoloTargetTracker,
    YoloTargetTrackerConfig,
)

__all__ = (
    "BaseTargetTracker",
    "CsrtTargetTracker",
    "CsrtTargetTrackerConfig",
    "IrstContrastTargetTracker",
    "IrstContrastTargetTrackerConfig",
    "TemplatePointTargetTracker",
    "TemplatePointTargetTrackerConfig",
    "YoloTargetTracker",
    "YoloTargetTrackerConfig",
)
