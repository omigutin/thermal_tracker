"""Источник кадров из папки с изображениями.

Полезен, когда видео уже нарезано на кадры, и мы хотим:
- детально отлаживать поведение на отдельных изображениях;
- сравнивать трекер на фиксированной последовательности;
- не зависеть от кодеков контейнера.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base_frame_reader import BaseFrameSource


class ImageSequenceFrameSource(BaseFrameSource):
    """Читает кадры из папки с изображениями по имени файла."""

    implementation_name = "image_sequence"
    is_ready = True

    def __init__(self, directory: str, patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp")) -> None:
        self.directory = Path(directory)
        self._files: list[Path] = []
        for pattern in patterns:
            self._files.extend(sorted(self.directory.glob(pattern)))
        self._index = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._files):
            return False, None

        path = self._files[self._index]
        self._index += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return (frame is not None), frame

    def close(self) -> None:
        self._files = []
