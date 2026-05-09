"""Сценарий NN для режима "кликнули по цели и ведём её".

Текущая первая реализация построена вокруг:
- YOLO как detector;
- внешнего tracker-конфига вроде ByteTrack;
- выбора одной цели поверх общего списка track id.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerPreset, build_preset
from ..domain.models import DetectedObject, GlobalMotion, ProcessedFrame, TrackSnapshot
from ..domain.runtime import ScenarioStepResult, SessionRuntimeState
from ..processing_stages.frame_preprocessing import FramePreprocessorManager
from ..processing_stages.frame_stabilization import FrameStabilizerManager
from ..processing_stages.target_tracking import TargetTrackerManager


class ManualClickNeuralPipeline:
    """Собирает рабочий NN-контур сопровождения одной цели."""

    def __init__(self, preset_name: str, preset_override: TrackerPreset | None = None) -> None:
        self.preset: TrackerPreset = preset_override or build_preset(preset_name)
        if self.preset.neural is None:
            raise RuntimeError(f"Пресет {preset_name!r} не содержит секцию [neural].")

        self.preprocessor = FramePreprocessorManager(self.preset.preprocessing.methods, self.preset.preprocessing)
        self.motion_estimator = FrameStabilizerManager(self.preset.global_motion.method, self.preset.global_motion)
        self.tracker = TargetTrackerManager(
            self.preset.tracker.method,
            self.preset.tracker,
            self.preset.click_selection,
            self.preset.neural,
        )

        self.current_frame: ProcessedFrame | None = None
        self.current_snapshot: TrackSnapshot = self.tracker.snapshot(GlobalMotion())

    @property
    def preset_name(self) -> str:
        return self.preset.name

    @property
    def candidate_objects(self) -> tuple[DetectedObject, ...]:
        """Возвращает список текущих детекций, которые можно показать в GUI."""

        return self.tracker.latest_detections

    def process_next_raw_frame(
        self,
        raw_frame: np.ndarray,
        runtime: SessionRuntimeState,
    ) -> ScenarioStepResult:
        self.current_frame = self.preprocessor.process(raw_frame)
        self.current_snapshot = self._process_runtime_actions(runtime)
        return ScenarioStepResult(frame=self.current_frame, snapshot=self.current_snapshot)

    def apply_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        if self.current_frame is None:
            return self.current_snapshot
        self.current_snapshot = self._process_static_actions(runtime)
        return self.current_snapshot

    def _process_runtime_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        assert self.current_frame is not None

        if runtime.reset_requested:
            snapshot = self.tracker.reset()
            runtime.reset_requested = False
        else:
            snapshot = self.current_snapshot

        motion = self.motion_estimator.estimate(self.current_frame)
        snapshot = self.tracker.update(self.current_frame, motion)

        if runtime.pending_click is not None:
            snapshot = self.tracker.start_tracking(self.current_frame, runtime.pending_click)
            runtime.pending_click = None

        return snapshot

    def _process_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        assert self.current_frame is not None
        snapshot = self.current_snapshot

        if runtime.reset_requested:
            snapshot = self.tracker.reset()
            runtime.reset_requested = False

        if runtime.pending_click is not None:
            snapshot = self.tracker.start_tracking(self.current_frame, runtime.pending_click)
            runtime.pending_click = None

        return snapshot
