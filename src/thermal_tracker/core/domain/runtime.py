"""Состояние выполнения и результаты шага сценария."""

from __future__ import annotations

from dataclasses import dataclass

from .models import DetectedObject, GlobalMotion, MotionDetectionResult, ProcessedFrame, TrackSnapshot


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
    global_motion: GlobalMotion
    motion_result: MotionDetectionResult
    raw_objects: list[DetectedObject]
    filtered_objects: list[DetectedObject]
