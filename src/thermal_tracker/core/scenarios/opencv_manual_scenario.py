"""
Пайплайн текущего этапа: ручной выбор цели и сопровождение одной цели:
- оператор кликает по объекту;
- система сама строит стартовый bbox;
- затем цель ведётся;
- при затянувшейся потере pipeline дёргает стадию target_recovery,
  которая пытается вернуть цель и при успехе пере-инициализирует трекер.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerPreset, build_preset
from ..domain.models import DetectedObject, GlobalMotion, ProcessedFrame, TrackSnapshot, TrackerState
from ..domain.runtime import ScenarioStepResult, SessionRuntimeState
from ..processing_stages.frame_preprocessing import FramePreprocessorManager
from ..processing_stages.frame_stabilization import FrameStabilizerManager
from ..processing_stages.target_recovery import TargetRecovererManager
from ..processing_stages.target_tracking import TargetTrackerManager


class ManualClickTrackingPipeline:
    """Склеивает рабочие этапы текущего трекера в один понятный контур."""

    def __init__(self, preset_name: str, preset_override: TrackerPreset | None = None) -> None:
        self.preset: TrackerPreset = preset_override or build_preset(preset_name)
        self.preprocessor = FramePreprocessorManager(self.preset.preprocessing.methods, self.preset.preprocessing)
        self.motion_estimator = FrameStabilizerManager(self.preset.global_motion.method, self.preset.global_motion)
        self.tracker = TargetTrackerManager(
            self.preset.tracker.method,
            self.preset.tracker,
            self.preset.click_selection,
            self.preset.neural,
        )
        self.recoverer = TargetRecovererManager(
            self.preset.target_recovery.method,
            self.preset.target_recovery,
        )

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
            self.recoverer.reset()
            runtime.reset_requested = False
        else:
            snapshot = self.current_snapshot

        if runtime.pending_click is not None:
            snapshot = self.tracker.start_tracking(self.current_frame, runtime.pending_click)
            self.recoverer.reset()
            runtime.pending_click = None
            self.motion_estimator.estimate(self.current_frame)
            return self._after_tracker_step(snapshot)

        motion = self.motion_estimator.estimate(self.current_frame)
        snapshot = self.tracker.update(self.current_frame, motion)
        snapshot = self._maybe_recover(snapshot, motion)
        return self._after_tracker_step(snapshot)

    def _process_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        """Обрабатывает действия пользователя без продвижения видео вперёд."""

        assert self.current_frame is not None
        snapshot = self.current_snapshot

        if runtime.reset_requested:
            snapshot = self.tracker.reset()
            self.recoverer.reset()
            runtime.reset_requested = False

        if runtime.pending_click is not None:
            snapshot = self.tracker.start_tracking(self.current_frame, runtime.pending_click)
            self.recoverer.reset()
            runtime.pending_click = None

        return snapshot

    def _maybe_recover(self, snapshot: TrackSnapshot, motion: GlobalMotion) -> TrackSnapshot:
        """Дёргает стадию target_recovery, если цель уверенно потеряна.

        Recoverer вызывается только в SEARCHING и только когда счётчик
        потерянных кадров достиг порога. Если recoverer вернул bbox,
        трекер перезапускается через start_tracking — track_id сменится,
        что на текущем подэтапе нормально.
        """

        assert self.current_frame is not None

        if snapshot.state != TrackerState.SEARCHING:
            return snapshot
        if snapshot.lost_frames < self.preset.target_recovery.min_lost_frames:
            return snapshot
        if snapshot.predicted_bbox is None:
            return snapshot

        try:
            new_bbox = self.recoverer.reacquire(self.current_frame, snapshot.predicted_bbox, motion)
        except NotImplementedError:
            # Выбранный recoverer-метод пока без реализации.
            # Не рушим pipeline, продолжаем с текущим snapshot трекера.
            return snapshot

        if new_bbox is None:
            return snapshot

        center_x, center_y = new_bbox.center
        recovered_snapshot = self.tracker.start_tracking(
            self.current_frame,
            (int(round(center_x)), int(round(center_y))),
        )
        return recovered_snapshot

    def _after_tracker_step(self, snapshot: TrackSnapshot) -> TrackSnapshot:
        """Освежает память recoverer-а на каждом подтверждённом TRACKING-кадре."""

        if (
            self.current_frame is not None
            and snapshot.state == TrackerState.TRACKING
            and snapshot.bbox is not None
        ):
            self.recoverer.remember(self.current_frame, snapshot.bbox)
        return snapshot
