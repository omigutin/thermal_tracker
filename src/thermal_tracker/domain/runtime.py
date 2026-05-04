"""Runtime state and scenario step results."""

from __future__ import annotations

from dataclasses import dataclass

from .models import DetectedObject, GlobalMotion, MotionDetectionResult, ProcessedFrame, TrackSnapshot


@dataclass
class SessionRuntimeState:
    """Mutable user/session commands consumed while frames are processed."""

    pending_click: tuple[int, int] | None = None
    reset_requested: bool = False
    paused: bool = False
    step_once: bool = False


@dataclass
class ScenarioStepResult:
    """Result returned by an interactive scenario after one frame."""

    frame: ProcessedFrame
    snapshot: TrackSnapshot


@dataclass
class AutoScenarioStepResult:
    """Result returned by an automatic scenario after one frame."""

    frame: ProcessedFrame
    global_motion: GlobalMotion
    motion_result: MotionDetectionResult
    raw_objects: list[DetectedObject]
    filtered_objects: list[DetectedObject]
