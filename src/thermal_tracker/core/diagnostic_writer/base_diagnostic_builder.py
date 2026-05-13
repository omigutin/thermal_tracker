"""Базовый контракт сборщика диагностики кадра."""
from __future__ import annotations
from abc import ABC, abstractmethod
from .click_diagnostics import ClickDiagnostics
from .detection_diagnostics import DetectionDiagnostics
from .frame_diagnostics import FrameDiagnostics
from .preprocessing_diagnostics import PreprocessingDiagnostics
from .recovery_diagnostics import RecoveryDiagnostics
from .tracker_diagnostics import TrackerUpdateDiagnostics

class BaseDiagnosticBuilder(ABC):
    """Интерфейс для накопления диагностики в течение обработки кадра."""
    @abstractmethod
    def start_frame(self, frame_id: int) -> None: """Инициализирует сбор диагностики для нового кадра."""
    @abstractmethod
    def set_click(self, click: ClickDiagnostics) -> None: """Сохраняет диагностику этапа клика."""
    @abstractmethod
    def set_preprocessing(self, preprocessing: PreprocessingDiagnostics) -> None: """Сохраняет диагностику предпреобразования."""
    @abstractmethod
    def set_detection(self, detection: DetectionDiagnostics) -> None: """Сохраняет диагностику детекции."""
    @abstractmethod
    def set_tracker_update(self, tracker_update: TrackerUpdateDiagnostics) -> None: """Сохраняет диагностику обновления трекера."""
    @abstractmethod
    def set_recovery(self, recovery: RecoveryDiagnostics) -> None: """Сохраняет диагностику восстановления трека."""
    @abstractmethod
    def finalize_frame(self) -> FrameDiagnostics | None: """Возвращает диагностику и очищает состояние."""
