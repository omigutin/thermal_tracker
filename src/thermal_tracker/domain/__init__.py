"""Общие сущности нового контура трекинга.

Здесь живут вещи, которые не принадлежат одному конкретному шагу пайплайна:
- модели данных;
- простые контракты между этапами;
- небольшое runtime-состояние сессии.
"""

from .contracts import (
    ClickInitializer,
    FalseTargetFilter,
    FramePreprocessor,
    FrameRenderer,
    GlobalMotionEstimator,
    MotionDetector,
    ObjectBuilder,
    Reacquirer,
    SingleTargetTracker,
    VideoSource,
)
from .models import (
    BoundingBox,
    DetectedObject,
    GlobalMotion,
    MotionDetectionResult,
    ProcessedFrame,
    SelectionResult,
    TrackSnapshot,
    TrackerState,
)
from .runtime import AutoScenarioStepResult, ScenarioStepResult, SessionRuntimeState

__all__ = [
    "AutoScenarioStepResult",
    "BoundingBox",
    "ClickInitializer",
    "DetectedObject",
    "FalseTargetFilter",
    "FramePreprocessor",
    "FrameRenderer",
    "GlobalMotion",
    "GlobalMotionEstimator",
    "MotionDetectionResult",
    "MotionDetector",
    "ObjectBuilder",
    "ScenarioStepResult",
    "ProcessedFrame",
    "Reacquirer",
    "SelectionResult",
    "SessionRuntimeState",
    "SingleTargetTracker",
    "TrackSnapshot",
    "TrackerState",
    "VideoSource",
]
