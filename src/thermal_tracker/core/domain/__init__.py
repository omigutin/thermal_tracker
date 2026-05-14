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
    MotionLocalizationResult,
    ProcessedFrame,
    TrackerState,
)
from ..stages.target_tracking.result import TargetTrackingResult
from ..stages.target_selection.result import TargetSelectorResult
from ..stages.candidate_formation.result import CandidateFormerResult
from ..stages.frame_stabilization.result import FrameStabilizerResult
from .runtime import AutoScenarioStepResult, ScenarioStepResult, SessionRuntimeState

__all__ = [
    "AutoScenarioStepResult",
    "BoundingBox",
    "ClickInitializer",
    "FalseTargetFilter",
    "FramePreprocessor",
    "FrameRenderer",
    "GlobalMotionEstimator",
    "MotionLocalizerResult",
    "MotionDetector",
    "ObjectBuilder",
    "ScenarioStepResult",
    "ProcessedFrame",
    "Reacquirer",
    "SessionRuntimeState",
    "SingleTargetTracker",
    "TrackerState",
    "VideoSource",
]
