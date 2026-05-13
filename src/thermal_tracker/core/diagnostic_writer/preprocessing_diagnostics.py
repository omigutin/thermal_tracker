"""Диагностика предпреобразования теплового кадра."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PreprocessingDiagnostics:
    """Метрики качества кадра перед детекцией."""
    sharpness: float | None = None
    sharpness_baseline: float | None = None
    blur_active: bool | None = None
