"""Чтение видео через `cv2.VideoCapture`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ...errors import VideoOpenError
from .base_frame_reader import BaseFrameSource


class OpenCvVideoSource(BaseFrameSource):
    implementation_name = "opencv_video"
    is_ready = True
    """Простой источник кадров из видеофайла."""

    def __init__(self, video_path: str) -> None:
        self.video_path = str(Path(video_path))
        self._capture = cv2.VideoCapture(self.video_path)
        if not self._capture.isOpened():
            raise VideoOpenError(f"Could not open video: {self.video_path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Читает следующий кадр из файла."""

        ok, frame = self._capture.read()
        return ok, frame if ok else None

    @property
    def fps(self) -> float:
        """Возвращает FPS файла, а при сомнениях подставляет разумный минимум."""

        value = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 0 else 25.0

    @property
    def frame_count(self) -> int:
        """Возвращает общее число кадров, если OpenCV смог его определить."""

        value = int(round(float(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        return max(0, value)

    @property
    def current_frame_index(self) -> int:
        """Возвращает индекс последнего уже прочитанного кадра."""

        position = int(round(float(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)))
        return max(0, position - 1)

    def seek_frame(self, frame_index: int) -> None:
        """Перемещает указатель чтения к нужному кадру."""

        safe_index = max(0, int(frame_index))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, safe_index)

    def close(self) -> None:
        """Освобождает файл и внутренние ресурсы OpenCV."""

        self._capture.release()
