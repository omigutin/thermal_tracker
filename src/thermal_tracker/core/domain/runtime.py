"""Состояние выполнения и результаты шага сценария."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProcessedFrame, TrackSnapshot
from ..stages.candidate_formation.result import DetectedObject
from ..stages.frame_stabilization.result import FrameStabilizerResult
from ..stages.motion_localization import MotionLocalizerResult


@dataclass
class SessionRuntimeState:
    """Изменяемые команды пользователя или сессии, потребляемые при обработке кадров."""

    pending_click: tuple[int, int] | None = None
    reset_requested: bool = False
    paused: bool = False
    step_once: bool = False


@dataclass
class ScenarioStepResult:
    """Результат интерактивного сценария после обработки одного кадра."""

    frame: ProcessedFrame
    snapshot: TrackSnapshot


@dataclass
class AutoScenarioStepResult:
    """Результат автоматического сценария после обработки одного кадра."""

    frame: ProcessedFrame
    global_motion: FrameStabilizerResult
    motion_result: MotionLocalizerResult
    raw_objects: list[DetectedObject]
    filtered_objects: list[DetectedObject]
