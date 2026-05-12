"""Создание сценариев трекинга по имени из конфига."""

from __future__ import annotations

from collections.abc import Callable

from ..config import TrackerPreset
from .nn_auto_scenario import AutoNeuralDetectionPipeline
from .nn_manual_scenario import ManualClickNeuralPipeline
from .opencv_auto_motion_scenario import AutoMotionTrackingPipeline
from .opencv_manual_scenario import ManualClickTrackingPipeline

ScenarioBuilder = Callable[[str, TrackerPreset | None], object]

DEFAULT_PRESET_BY_SCENARIO = {
    "nn_manual": "yolo_general",
    "nn_auto": "yolo_auto",
    "opencv_manual": "opencv_general",
    "opencv_auto_motion": "opencv_general",
}

PIPELINE_KIND_TO_SCENARIO = {
    "manual_click_classical": "opencv_manual",
    "manual_click_neural": "nn_manual",
    "auto_neural_detection": "nn_auto",
    "auto_motion_tracking": "opencv_auto_motion",
}


def default_preset_for_scenario(scenario_name: str) -> str:
    normalized = normalize_scenario_name(scenario_name)
    return DEFAULT_PRESET_BY_SCENARIO.get(normalized, "opencv_general")


def normalize_scenario_name(scenario_name: str) -> str:
    """Приводит имя сценария или псевдоним pipeline_kind к каноническому имени сценария."""

    requested = (scenario_name or "").strip()
    return PIPELINE_KIND_TO_SCENARIO.get(requested, requested or "opencv_manual")


def _build_opencv_manual(preset_name: str, preset_override: TrackerPreset | None) -> object:
    return ManualClickTrackingPipeline(preset_name, preset_override=preset_override)


def _build_nn_manual(preset_name: str, preset_override: TrackerPreset | None) -> object:
    return ManualClickNeuralPipeline(preset_name, preset_override=preset_override)


def _build_nn_auto(preset_name: str, preset_override: TrackerPreset | None) -> object:
    return AutoNeuralDetectionPipeline(preset_name, preset_override=preset_override)


def _build_opencv_auto_motion(preset_name: str, preset_override: TrackerPreset | None) -> object:
    return AutoMotionTrackingPipeline(preset_override.name if preset_override is not None else preset_name)


class ScenarioFactory:
    _builders: dict[str, ScenarioBuilder] = {
        "opencv_manual": _build_opencv_manual,
        "nn_manual": _build_nn_manual,
        "nn_auto": _build_nn_auto,
        "opencv_auto_motion": _build_opencv_auto_motion,
    }

    @classmethod
    def create(
        cls,
        scenario_name: str,
        *,
        preset_name: str | None = None,
        preset_override: TrackerPreset | None = None,
    ) -> object:
        normalized = normalize_scenario_name(scenario_name)
        try:
            builder = cls._builders[normalized]
        except KeyError as exc:
            known = ", ".join(sorted(cls._builders))
            raise ValueError(f"Unknown scenario {scenario_name!r}. Known scenarios: {known}.") from exc

        resolved_preset = preset_name or (
            preset_override.name if preset_override is not None else default_preset_for_scenario(normalized)
        )
        return builder(resolved_preset, preset_override)

    @classmethod
    def create_from_preset(cls, preset: TrackerPreset) -> object:
        scenario_name = PIPELINE_KIND_TO_SCENARIO.get(preset.pipeline_kind, "opencv_manual")
        return cls.create(scenario_name, preset_name=preset.name, preset_override=preset)
