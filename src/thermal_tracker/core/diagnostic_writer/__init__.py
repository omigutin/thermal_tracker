"""Фабрика и экспорт сущностей диагностики runtime-кадра."""
from __future__ import annotations
from .base_diagnostic_builder import BaseDiagnosticBuilder
from .click_diagnostics import ClickDiagnostics
from .detection_diagnostics import CandidateInfo, DetectionDiagnostics
from .diagnostic_builder import DiagnosticBuilder
from .frame_diagnostics import FrameDiagnostics
from .no_diagnostic_builder import NoDiagnosticBuilder
from .preprocessing_diagnostics import PreprocessingDiagnostics
from .recovery_diagnostics import RecoveryDiagnostics
from .tracker_diagnostics import TrackerUpdateDiagnostics

def create_diagnostic_builder(enabled: bool) -> BaseDiagnosticBuilder:
    """Создает реализацию билдера под режим диагностики."""
    return DiagnosticBuilder() if enabled else NoDiagnosticBuilder()
