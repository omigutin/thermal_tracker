"""Заготовка под мультикамерный источник.

Этот класс нужен на вырост:
- позже система должна уметь жить рядом с несколькими камерами;
- уже сейчас полезно показать, что архитектура не привязана к одному файлу.

Пока это сознательная заглушка, а не недоделка.
"""

from __future__ import annotations

from .base_frame_reader import BaseFrameSource


class MultiCameraFrameSource(BaseFrameSource):
    """Будущий агрегатор нескольких синхронных источников кадров."""

    implementation_name = "multi_camera"
    is_ready = False

    def __init__(self, source_configs: list[dict[str, object]]) -> None:
        self.source_configs = source_configs

    def read(self):
        raise NotImplementedError("Мультикамерный источник пока не реализован.")

    def close(self) -> None:
        """Пока освобождать нечего."""
