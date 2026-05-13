"""Реальная реализация сборщика диагностики кадра."""
from __future__ import annotations
from .base_diagnostic_builder import BaseDiagnosticBuilder
from .click_diagnostics import ClickDiagnostics
from .detection_diagnostics import DetectionDiagnostics
from .frame_diagnostics import FrameDiagnostics
from .preprocessing_diagnostics import PreprocessingDiagnostics
from .recovery_diagnostics import RecoveryDiagnostics
from .tracker_diagnostics import TrackerUpdateDiagnostics

class DiagnosticBuilder(BaseDiagnosticBuilder):
    """Копит данные этапов и формирует FrameDiagnostics."""
    def __init__(self) -> None:
        self._current_frame_id: int | None = None
        self._click: ClickDiagnostics | None = None
        self._preprocessing: PreprocessingDiagnostics | None = None
        self._detection: DetectionDiagnostics | None = None
        self._tracker_update: TrackerUpdateDiagnostics | None = None
        self._recovery: RecoveryDiagnostics | None = None
    def start_frame(self, frame_id: int) -> None:
        self._current_frame_id = frame_id
        self._click = self._preprocessing = self._detection = self._tracker_update = self._recovery = None
    def set_click(self, click: ClickDiagnostics) -> None: self._click = click
    def set_preprocessing(self, preprocessing: PreprocessingDiagnostics) -> None: self._preprocessing = preprocessing
    def set_detection(self, detection: DetectionDiagnostics) -> None: self._detection = detection
    def set_tracker_update(self, tracker_update: TrackerUpdateDiagnostics) -> None: self._tracker_update = tracker_update
    def set_recovery(self, recovery: RecoveryDiagnostics) -> None: self._recovery = recovery
    def finalize_frame(self) -> FrameDiagnostics | None:
        if self._current_frame_id is None:
            return None
        result = FrameDiagnostics(self._current_frame_id, self._click, self._preprocessing, self._detection, self._tracker_update, self._recovery)
        self._current_frame_id = None
        self._click = self._preprocessing = self._detection = self._tracker_update = self._recovery = None
        return result
