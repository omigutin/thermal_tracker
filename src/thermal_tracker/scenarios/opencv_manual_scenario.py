"""
Пайплайн текущего этапа: ручной выбор цели и сопровождение одной цели:
- оператор кликает по объекту;
- система сама строит стартовый bbox;
- затем цель ведётся и при потере пытается восстановиться.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerPreset, build_preset
from ..domain.models import DetectedObject, GlobalMotion, ProcessedFrame, TrackSnapshot
from ..domain.runtime import ScenarioStepResult, SessionRuntimeState
from ..processing_stages.frame_preprocessing import ThermalFramePreprocessor
from ..processing_stages.frame_stabilization import PhaseCorrelationMotionEstimator
from ..processing_stages.target_tracking import ClickToTrackSingleTargetTracker


class ManualClickTrackingPipeline:
    """Склеивает рабочие этапы текущего трекера в один понятный контур."""

    def __init__(self, preset_name: str, preset_override: TrackerPreset | None = None) -> None:
        self.preset: TrackerPreset = preset_override or build_preset(preset_name)
        self.preprocessor = ThermalFramePreprocessor(self.preset.preprocessing)
        self.motion_estimator = PhaseCorrelationMotionEstimator(self.preset.global_motion)
        self.tracker = ClickToTrackSingleTargetTracker(self.preset.tracker, self.preset.click_selection)

        self.current_frame: ProcessedFrame | None = None
        self.current_snapshot: TrackSnapshot = self.tracker.snapshot(GlobalMotion())

    @property
    def preset_name(self) -> str:
        """Короткое имя активного пресета."""

        return self.preset.name

    @property
    def candidate_objects(self) -> tuple[DetectedObject, ...]:
        """Классический pipeline не хранит список отдельных детекций для GUI."""

        return ()

    def process_next_raw_frame(
        self,
        raw_frame: np.ndarray,
        runtime: SessionRuntimeState,
    ) -> ScenarioStepResult:
        """Обрабатывает новый сырой кадр и обновляет состояние трека."""

        self.current_frame = self.preprocessor.process(raw_frame)
        self.current_snapshot = self._process_runtime_actions(runtime)
        return ScenarioStepResult(frame=self.current_frame, snapshot=self.current_snapshot)

    def apply_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        """Обрабатывает клик или сброс, когда видео стоит на паузе."""

        if self.current_frame is None:
            return self.current_snapshot
        self.current_snapshot = self._process_static_actions(runtime)
        return self.current_snapshot

    def _process_runtime_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        """Разбирает клик, сброс и обычное обновление на новом кадре."""

        assert self.current_frame is not None

        if runtime.reset_requested:
            snapshot = self.tracker.reset()
            runtime.reset_requested = False
        else:
            snapshot = self.current_snapshot

        if runtime.pending_click is not None:
            snapshot = self.tracker.start_tracking(self.current_frame, runtime.pending_click)
            runtime.pending_click = None
            self.motion_estimator.estimate(self.current_frame)
            return snapshot

        motion = self.motion_estimator.estimate(self.current_frame)
        return self.tracker.update(self.current_frame, motion)

    def _process_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        """Обрабатывает действия пользователя без продвижения видео вперёд."""

        assert self.current_frame is not None
        snapshot = self.current_snapshot

        if runtime.reset_requested:
            snapshot = self.tracker.reset()
            runtime.reset_requested = False

        if runtime.pending_click is not None:
            snapshot = self.tracker.start_tracking(self.current_frame, runtime.pending_click)
            runtime.pending_click = None

        return snapshot
