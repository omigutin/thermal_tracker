"""
Пайплайн ручного клика и сопровождения одной цели:
- оператор кликает по объекту;
- система сама строит стартовый bbox;
- цель ведётся;
- при потере pipeline дёргает стадию target_recovery и через
  track-before-confirm возвращает трек под тем же track_id;
- если recovery в окне не подтверждается, pipeline уходит в LOST.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerPreset, build_preset
from ..stages.target_tracking.target_tracker_type import TargetTrackerType
from ..domain.models import BoundingBox, DetectedObject, GlobalMotion, ProcessedFrame, TrackSnapshot, TrackerState
from ..domain.runtime import ScenarioStepResult, SessionRuntimeState
from ..stages.frame_preprocessing import FramePreprocessorManager
from ..stages.frame_stabilization import FrameStabilizerManager
from ..stages.target_recovery import TargetRecovererManager
from ..stages.target_tracking import TargetTrackerManager
from ..state_machine import StateMachine


class ManualClickTrackingPipeline:
    """Склеивает рабочие этапы текущего трекера в один понятный контур."""

    def __init__(self, preset_name: str, preset_override: TrackerPreset | None = None) -> None:
        self.preset: TrackerPreset = preset_override or build_preset(preset_name)

        # Выбираем трекер: IRST имеет приоритет, если задана секция [irst_tracking].
        if self.preset.irst_tracker is not None:
            tracker_type = TargetTrackerType.IRST_CONTRAST
            tracker_config = self.preset.irst_tracker
            # IRST-трекер управляет IDLE через max_lost_frames — граница кадра не проверяется.
            self._edge_exit_margin: int = 0
        elif self.preset.opencv_tracker is not None:
            tracker_type = TargetTrackerType.OPENCV_TEMPLATE_POINT
            tracker_config = self.preset.opencv_tracker
            self._edge_exit_margin = int(self.preset.opencv_tracker.edge_exit_margin)
        else:
            raise RuntimeError(
                f"Preset {self.preset.name!r} does not contain [opencv_tracking] or [irst_tracking] section "
                f"required by ManualClickTrackingPipeline."
            )

        self.preprocessor = FramePreprocessorManager(self.preset.preprocessing.methods, self.preset.preprocessing)
        self.motion_estimator = FrameStabilizerManager(self.preset.global_motion.method, self.preset.global_motion)
        self.tracker = TargetTrackerManager(
            tracker_type,
            tracker_config,
            self.preset.click_selection,
        )
        self.recoverer = TargetRecovererManager(
            self.preset.target_recovery.method,
            self.preset.target_recovery,
        )

        self.current_frame: ProcessedFrame | None = None
        self.current_snapshot: TrackSnapshot = self.tracker.snapshot(GlobalMotion())

        # Pipeline-уровневая state machine. Не имеет таблицы переходов:
        # бизнес-правила переходов живут в коде ниже, machine хранит current.
        self._state_machine: StateMachine[TrackerState] = StateMachine(TrackerState.IDLE)

        # Track id, под которым ведём трек снаружи. Сохраняется через recovery.
        self._tracking_track_id: int | None = None

        # Последний достоверный TRACKING-bbox. Нужен, чтобы pipeline мог
        # отказаться вызывать recoverer, если цель ушла за границу кадра.
        self._last_good_bbox: BoundingBox | None = None

        # Pending-recovery: рабочее состояние track-before-confirm.
        self._pending_bbox: BoundingBox | None = None
        self._pending_confirmations: int = 0
        self._pending_age_frames: int = 0

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
            self._on_reset()
            runtime.reset_requested = False

        if runtime.pending_click is not None:
            click_point = runtime.pending_click
            runtime.pending_click = None
            snapshot = self._on_click(self.current_frame, click_point)
            self.motion_estimator.estimate(self.current_frame)
            return snapshot

        motion = self.motion_estimator.estimate(self.current_frame)
        return self._step_with_motion(self.current_frame, motion)

    def _process_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        """Обрабатывает действия пользователя без продвижения видео вперёд."""

        assert self.current_frame is not None
        snapshot = self.current_snapshot

        if runtime.reset_requested:
            snapshot = self._on_reset()
            runtime.reset_requested = False

        if runtime.pending_click is not None:
            click_point = runtime.pending_click
            runtime.pending_click = None
            snapshot = self._on_click(self.current_frame, click_point)

        return snapshot

    def _on_reset(self) -> TrackSnapshot:
        """Полный сброс: трекер, recoverer, pending-recovery, state machine."""

        snapshot = self.tracker.reset()
        self.recoverer.reset()
        self._reset_pending_recovery()
        self._tracking_track_id = None
        self._last_good_bbox = None
        self._state_machine.reset(TrackerState.IDLE)
        return snapshot

    def _on_click(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
        """Старт нового трека по клику пользователя."""

        snapshot = self.tracker.start_tracking(frame, point)
        self.recoverer.reset()
        self._reset_pending_recovery()
        self._last_good_bbox = None
        self._tracking_track_id = snapshot.track_id
        self._state_machine.transition_to(
            TrackerState.TRACKING if snapshot.state == TrackerState.TRACKING else TrackerState.IDLE
        )
        return self._after_tracker_step(snapshot)

    def _step_with_motion(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        """Один шаг pipeline: трекер обновляется, pipeline решает state."""

        if self._state_machine.current == TrackerState.RECOVERING:
            return self._handle_recovering(frame, motion)

        snapshot = self.tracker.update(frame, motion)
        return self._handle_tracker_snapshot(snapshot, motion)

    def _handle_tracker_snapshot(self, snapshot: TrackSnapshot, motion: GlobalMotion) -> TrackSnapshot:
        """Принимает snapshot трекера и решает pipeline-уровневый переход."""

        if snapshot.state == TrackerState.TRACKING:
            if self._state_machine.current != TrackerState.TRACKING:
                self._state_machine.transition_to(TrackerState.TRACKING)
            return self._after_tracker_step(snapshot)

        if snapshot.state == TrackerState.SEARCHING:
            if self._state_machine.current != TrackerState.SEARCHING:
                self._state_machine.transition_to(TrackerState.SEARCHING)
            return self._maybe_start_recovery(snapshot, motion)

        # Трекер сам ушёл в IDLE: пробуем интерпретировать как LOST,
        # если до этого уже искали; иначе оставляем IDLE.
        if self._state_machine.current in (TrackerState.SEARCHING, TrackerState.RECOVERING, TrackerState.TRACKING):
            self._state_machine.transition_to(TrackerState.LOST)
            self._reset_pending_recovery()
            self._tracking_track_id = None
            self._last_good_bbox = None
        return snapshot

    def _maybe_start_recovery(self, snapshot: TrackSnapshot, motion: GlobalMotion) -> TrackSnapshot:
        """Из SEARCHING пытается стартовать pending recovery."""

        if snapshot.lost_frames < self.preset.target_recovery.min_lost_frames:
            return snapshot
        if snapshot.predicted_bbox is None:
            return snapshot
        if self._last_good_bbox is not None and self._was_at_frame_border(self._last_good_bbox):
            return snapshot

        candidate = self._reacquire_safely(snapshot.predicted_bbox, motion, snapshot.lost_frames)
        if candidate is None:
            return snapshot

        self._pending_bbox = candidate
        self._pending_confirmations = 1
        self._pending_age_frames = 1
        self._state_machine.transition_to(TrackerState.RECOVERING)
        return self._make_recovering_snapshot(candidate, snapshot.lost_frames, motion)

    def _handle_recovering(self, frame: ProcessedFrame, motion: GlobalMotion) -> TrackSnapshot:
        """Каждый кадр в RECOVERING: пытается подтвердить или перейти в LOST."""

        assert self._pending_bbox is not None

        self._pending_age_frames += 1

        if self._pending_age_frames > self.preset.target_recovery.recovery_window_frames:
            return self._fall_to_lost(motion)

        candidate = self._reacquire_safely(self._pending_bbox, motion, self._pending_age_frames)
        if candidate is None:
            self._pending_confirmations = 0
            return self._make_recovering_snapshot(self._pending_bbox, self._pending_age_frames, motion)

        if not self._is_consistent(candidate, self._pending_bbox):
            self._pending_confirmations = 1
            self._pending_bbox = candidate
            return self._make_recovering_snapshot(candidate, self._pending_age_frames, motion)

        self._pending_bbox = candidate
        self._pending_confirmations += 1

        if self._pending_confirmations >= self.preset.target_recovery.confirm_frames:
            return self._promote_to_tracking(frame, candidate, motion)

        return self._make_recovering_snapshot(candidate, self._pending_age_frames, motion)

    def _promote_to_tracking(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        motion: GlobalMotion,
    ) -> TrackSnapshot:
        """Подтверждённый recovery: возобновляем трек с тем же track_id."""

        track_id = self._tracking_track_id if self._tracking_track_id is not None else 0
        snapshot = self.tracker.resume_tracking(frame, bbox, track_id)
        self._reset_pending_recovery()
        self._tracking_track_id = snapshot.track_id
        self._state_machine.transition_to(TrackerState.TRACKING)
        return self._after_tracker_step(snapshot)

    def _fall_to_lost(self, motion: GlobalMotion) -> TrackSnapshot:
        """Окно RECOVERING исчерпано — переходим в LOST и сбрасываем трекер."""

        self.tracker.reset()
        lost_snapshot = TrackSnapshot(
            state=TrackerState.LOST,
            track_id=self._tracking_track_id,
            bbox=None,
            predicted_bbox=None,
            search_region=None,
            score=0.0,
            lost_frames=self._pending_age_frames,
            global_motion=motion,
            message=f"Lost target #{self._tracking_track_id}",
        )
        self._reset_pending_recovery()
        self._tracking_track_id = None
        self._last_good_bbox = None
        self._state_machine.transition_to(TrackerState.LOST)
        return lost_snapshot

    def _make_recovering_snapshot(
        self,
        bbox: BoundingBox,
        lost_frames: int,
        motion: GlobalMotion,
    ) -> TrackSnapshot:
        """Снимок, который видит GUI/JSONL, пока идёт track-before-confirm."""

        return TrackSnapshot(
            state=TrackerState.RECOVERING,
            track_id=self._tracking_track_id,
            bbox=bbox,
            predicted_bbox=bbox,
            search_region=None,
            score=0.0,
            lost_frames=lost_frames,
            global_motion=motion,
            message=(
                f"Recovering target #{self._tracking_track_id} "
                f"({self._pending_confirmations}/{self.preset.target_recovery.confirm_frames})"
            ),
        )

    def _reacquire_safely(
        self,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
        lost_frames: int,
    ) -> BoundingBox | None:
        """Обёртка вокруг recoverer.reacquire с поглощением NotImplementedError."""

        assert self.current_frame is not None
        try:
            return self.recoverer.reacquire(self.current_frame, last_bbox, motion, lost_frames=lost_frames)
        except NotImplementedError:
            return None

    def _is_consistent(self, candidate: BoundingBox, reference: BoundingBox) -> bool:
        """Согласован ли новый recovery-bbox с предыдущим (центр близко)."""

        cx_new, cy_new = candidate.center
        cx_ref, cy_ref = reference.center
        distance = float(np.hypot(cx_new - cx_ref, cy_new - cy_ref))
        max_dim = max(reference.width, reference.height, 1)
        return distance <= max_dim * 1.5

    def _reset_pending_recovery(self) -> None:
        """Сбрасывает рабочее состояние track-before-confirm."""

        self._pending_bbox = None
        self._pending_confirmations = 0
        self._pending_age_frames = 0

    def _after_tracker_step(self, snapshot: TrackSnapshot) -> TrackSnapshot:
        """Освежает память recoverer-а на каждом подтверждённом TRACKING-кадре."""

        if (
            self.current_frame is not None
            and snapshot.state == TrackerState.TRACKING
            and snapshot.bbox is not None
        ):
            self.recoverer.remember(self.current_frame, snapshot.bbox)
            self._last_good_bbox = snapshot.bbox
        return snapshot

    def _was_at_frame_border(self, bbox: BoundingBox) -> bool:
        """Возвращает True, если bbox касался границы кадра в пределах edge_exit_margin."""

        if self.current_frame is None:
            return False
        margin = self._edge_exit_margin
        frame_h, frame_w = self.current_frame.bgr.shape[:2]
        return (
            bbox.x <= margin
            or bbox.y <= margin
            or bbox.x2 >= frame_w - margin
            or bbox.y2 >= frame_h - margin
        )
