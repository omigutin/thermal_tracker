from __future__ import annotations

from enum import StrEnum


class TargetRecovererType(StrEnum):
    """Типы операций повторного захвата цели."""

    # Ищет потерянную цель рядом с последним bbox по сохранённым шаблонам.
    LOCAL_TEMPLATE = "local_template"
    # Ищет маленькую тепловую цель по локальному контрасту в расширенной зоне.
    IRST_CONTRAST = "irst_contrast"
