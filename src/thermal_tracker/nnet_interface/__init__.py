"""Интерфейсный слой для подключения нейросетевых моделей."""

from .base_nnet_interface import BaseNnetInterface
from .yolo_nnet_interface import YoloNnetInterface

__all__ = ["BaseNnetInterface", "YoloNnetInterface"]
