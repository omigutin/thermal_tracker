"""Сборки готовых пайплайнов под разные сценарии."""

from .opencv_auto_motion_scenario import AutoMotionTrackingPipeline
from .nn_auto_scenario import AutoNeuralDetectionPipeline
from .nn_manual_scenario import ManualClickNeuralPipeline
from .opencv_manual_scenario import ManualClickTrackingPipeline
from .scenario_factory import ScenarioFactory, default_preset_for_scenario, normalize_scenario_name

__all__ = [
    "AutoMotionTrackingPipeline",
    "AutoNeuralDetectionPipeline",
    "ManualClickNeuralPipeline",
    "ManualClickTrackingPipeline",
    "ScenarioFactory",
    "default_preset_for_scenario",
    "normalize_scenario_name",
]
