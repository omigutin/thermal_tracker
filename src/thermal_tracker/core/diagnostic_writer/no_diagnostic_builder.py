"""No-op реализация сборщика диагностики."""
from __future__ import annotations
from .base_diagnostic_builder import BaseDiagnosticBuilder
from .click_diagnostics import ClickDiagnostics
from .detection_diagnostics import DetectionDiagnostics
from .frame_diagnostics import FrameDiagnostics
from .preprocessing_diagnostics import PreprocessingDiagnostics
from .recovery_diagnostics import RecoveryDiagnostics
from .tracker_diagnostics import TrackerUpdateDiagnostics

class NoDiagnosticBuilder(BaseDiagnosticBuilder):
    """Пустая реализация для режима с отключенной диагностикой."""
    def start_frame(self, frame_id: int) -> None: return None
    def set_click(self, click: ClickDiagnostics) -> None: return None
    def set_preprocessing(self, preprocessing: PreprocessingDiagnostics) -> None: return None
    def set_detection(self, detection: DetectionDiagnostics) -> None: return None
    def set_tracker_update(self, tracker_update: TrackerUpdateDiagnostics) -> None: return None
    def set_recovery(self, recovery: RecoveryDiagnostics) -> None: return None
    def finalize_frame(self) -> FrameDiagnostics | None: return None
