"""Движки инференса для нейросетевых пайплайнов."""

from .base_inference_engine import BaseInferenceEngine
from .ultralytics_yolo_engine import UltralyticsYoloEngine

__all__ = ["BaseInferenceEngine", "UltralyticsYoloEngine"]

