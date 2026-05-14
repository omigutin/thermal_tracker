"""Автоматический NN-сценарий без ручного клика.

Сценарий простой:
1. берём кадр;
2. при необходимости слегка уменьшаем его;
3. оцениваем движение камеры для статуса и будущего развития;
4. прогоняем кадр через нейросеть;
5. отдаём список найденных объектов в GUI.

Этот контур нужен, когда хочется просто открыть ролик и посмотреть,
что модель находит сама, без ручного выбора одной цели.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerPreset, build_preset
from ..domain.models import ProcessedFrame, TrackSnapshot, TrackerState
from ..stages.candidate_formation.result import DetectedObject
from ..stages.frame_stabilization.result import FrameStabilizerResult
from ..domain.runtime import ScenarioStepResult, SessionRuntimeState
from ..nnet_interface import YoloNnetInterface
from ..stages.frame_preprocessing import FramePreprocessorManager
from ..stages.frame_stabilization import FrameStabilizerManager


class AutoNeuralDetectionPipeline:
    """Автоматический режим, где нейросеть сама ищет объекты на каждом кадре."""

    def __init__(self, preset_name: str, preset_override: TrackerPreset | None = None) -> None:
        self.preset: TrackerPreset = preset_override or build_preset(preset_name)
        if self.preset.neural is None:
            raise RuntimeError(f"Пресет {preset_name!r} не содержит секцию [neural].")

        self.preprocessor = FramePreprocessorManager(self.preset.preprocessing.methods, self.preset.preprocessing)
        self.motion_estimator = FrameStabilizerManager(self.preset.global_motion.method, self.preset.global_motion)
        self.engine = YoloNnetInterface(self.preset.neural)

        self.current_frame: ProcessedFrame | None = None
        self.current_snapshot: TrackSnapshot = self._build_snapshot(FrameStabilizerResult(), (), "Нейросеть ещё не запускалась.")
        self._candidate_objects: tuple[DetectedObject, ...] = ()

    @property
    def preset_name(self) -> str:
        """Короткое имя активного пресета."""

        return self.preset.name

    @property
    def candidate_objects(self) -> tuple[DetectedObject, ...]:
        """Текущие объекты, найденные моделью на последнем кадре."""

        return self._candidate_objects

    def process_next_raw_frame(
        self,
        raw_frame: np.ndarray,
        runtime: SessionRuntimeState,
    ) -> ScenarioStepResult:
        """Прогоняет новый кадр через полностью автоматический NN-контур."""

        self.current_frame = self.preprocessor.process(raw_frame)
        motion = self.motion_estimator.apply(self.current_frame)
        self._candidate_objects = tuple(self.engine.track(self.current_frame.bgr))
        self.current_snapshot = self._build_snapshot(
            motion,
            self._candidate_objects,
            self._build_message(len(self._candidate_objects)),
        )
        self._consume_runtime_actions(runtime, motion)
        return ScenarioStepResult(frame=self.current_frame, snapshot=self.current_snapshot)

    def apply_static_actions(self, runtime: SessionRuntimeState) -> TrackSnapshot:
        """В авто-режиме клик не нужен, но честно объясняем это пользователю."""

        motion = self.current_snapshot.global_motion
        self._consume_runtime_actions(runtime, motion)
        return self.current_snapshot

    def _consume_runtime_actions(self, runtime: SessionRuntimeState, motion: FrameStabilizerResult) -> None:
        """Съедает клик и reset без побочных приключений."""

        if runtime.reset_requested:
            runtime.reset_requested = False
            self.current_snapshot = self._build_snapshot(
                motion,
                self._candidate_objects,
                "Авто-режим сброшен. Нейросеть продолжает сама искать все цели.",
            )

        if runtime.pending_click is not None:
            runtime.pending_click = None
            self.current_snapshot = self._build_snapshot(
                motion,
                self._candidate_objects,
                "В этом режиме клик не нужен: нейросеть сама показывает все найденные объекты.",
            )

    def _build_snapshot(
        self,
        motion: FrameStabilizerResult,
        detections: tuple[DetectedObject, ...],
        message: str,
    ) -> TrackSnapshot:
        """Собирает компактный снимок состояния для GUI."""

        score = max((detection.confidence for detection in detections), default=0.0)
        return TrackSnapshot(
            state=TrackerState.IDLE,
            track_id=None,
            bbox=None,
            predicted_bbox=None,
            search_region=None,
            score=score,
            lost_frames=0,
            global_motion=motion,
            message=message,
        )

    def _build_message(self, count: int) -> str:
        """Собирает короткое и честное сообщение для правой панели."""

        if count <= 0:
            return f"Авто-режим: модель {self.engine.mode_name} сейчас не нашла ни одной цели."
        if count == 1:
            return f"Авто-режим: найдена 1 цель через {self.engine.mode_name}."
        return f"Авто-режим: найдено целей: {count} через {self.engine.mode_name}."
