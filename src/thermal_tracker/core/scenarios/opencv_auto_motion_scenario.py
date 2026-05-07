"""Пайплайн автоматического режима.

Смысл этого контура такой:
1. кадр предобрабатывается;
2. оценивается движение камеры;
3. ищется движение на сцене;
4. из маски строятся объекты-кандидаты;
5. кандидаты фильтруются.

Сейчас это не полноценный multi-target manager, а честно подготовленный
автоматический режим детекта и первичной сборки объектов. Этого уже хватает,
чтобы показывать архитектуру и начинать расширять авто-контур дальше.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackerPreset, build_preset
from ..domain.models import DetectedObject, GlobalMotion, MotionDetectionResult, ProcessedFrame
from ..domain.runtime import AutoScenarioStepResult
from ..processing_stages.candidate_filtering import CandidateFilterManager
from ..processing_stages.moving_area_detection import MovingAreaDetectorManager
from ..processing_stages.target_candidate_extraction import TargetCandidateExtractorManager
from ..processing_stages.frame_preprocessing import FramePreprocessorManager
from ..processing_stages.frame_stabilization import FrameStabilizerManager


class AutoMotionTrackingPipeline:
    """Готовый автоматический контур для сцен без ручного клика."""

    def __init__(self, preset_name: str) -> None:
        self.preset: TrackerPreset = build_preset(preset_name)
        self.preprocessor = FramePreprocessorManager(self.preset.preprocessing.method, self.preset.preprocessing)
        self.motion_estimator = FrameStabilizerManager(self.preset.global_motion.method, self.preset.global_motion)
        self.motion_detector = MovingAreaDetectorManager(self.preset.moving_area_detection.method)
        self.object_builder = TargetCandidateExtractorManager(self.preset.target_candidate_extraction.method)
        self.false_target_filter = CandidateFilterManager(self.preset.candidate_filtering.filters)

        self.current_frame: ProcessedFrame | None = None
        self.current_motion: GlobalMotion = GlobalMotion()
        self.current_detection: MotionDetectionResult = MotionDetectionResult(
            mask=np.zeros((1, 1), dtype=np.uint8),
            source_name="empty",
        )
        self.current_raw_objects: list[DetectedObject] = []
        self.current_filtered_objects: list[DetectedObject] = []

    @property
    def preset_name(self) -> str:
        """Короткое имя активного пресета."""

        return self.preset.name

    def process_next_raw_frame(self, raw_frame: np.ndarray) -> AutoScenarioStepResult:
        """Прогоняет кадр через автоматический контур."""

        self.current_frame = self.preprocessor.process(raw_frame)
        self.current_motion = self.motion_estimator.estimate(self.current_frame)
        self.current_detection = self.motion_detector.detect(self.current_frame, self.current_motion)
        self.current_raw_objects = self.object_builder.build(self.current_frame, self.current_detection)
        self.current_filtered_objects = self.false_target_filter.filter(
            self.current_frame,
            self.current_raw_objects,
            self.current_motion,
        )
        return AutoScenarioStepResult(
            frame=self.current_frame,
            global_motion=self.current_motion,
            motion_result=self.current_detection,
            raw_objects=self.current_raw_objects,
            filtered_objects=self.current_filtered_objects,
        )
