"""Экспорт загрузчика алгоритмических пресетов."""

from .preset_OLD import (
    AVAILABLE_PRESETS,
    DEFAULT_PRESET_NAME,
    PresetPresentation,
    TrackerPreset,
    build_preset,
    get_available_preset_names,
    get_preset_presentation,
)

__all__ = [
    "AVAILABLE_PRESETS",
    "DEFAULT_PRESET_NAME",
    "PresetPresentation",
    "TrackerPreset",
    "build_preset",
    "get_available_preset_names",
    "get_preset_presentation",
]
