"""Контейнер полной диагностики одного кадра."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
from .click_diagnostics import ClickDiagnostics
from .detection_diagnostics import DetectionDiagnostics
from .preprocessing_diagnostics import PreprocessingDiagnostics
from .recovery_diagnostics import RecoveryDiagnostics
from .tracker_diagnostics import TrackerUpdateDiagnostics

@dataclass(frozen=True)
class FrameDiagnostics:
    """Собранная диагностика по всем этапам обработки кадра."""
    frame_id: int
    click: ClickDiagnostics | None = None
    preprocessing: PreprocessingDiagnostics | None = None
    detection: DetectionDiagnostics | None = None
    tracker_update: TrackerUpdateDiagnostics | None = None
    recovery: RecoveryDiagnostics | None = None

    def to_dict(self) -> dict[str, object]:
        """Преобразует объект в JSON-совместимый словарь."""
        return _to_builtin_types(asdict(self))

def _to_builtin_types(value: Any) -> Any:
    """Рекурсивно приводит numpy- и похожие скаляры к python-типам."""
    if isinstance(value, dict):
        return {str(key): _to_builtin_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin_types(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin_types(item) for item in value]
    scalar_to_python = getattr(value, "item", None)
    if callable(scalar_to_python):
        try:
            return scalar_to_python()
        except (TypeError, ValueError):
            return value
    return value
