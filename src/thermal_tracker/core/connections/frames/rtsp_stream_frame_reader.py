"""Заготовка под RTSP-поток.

Почему файл нужен уже сейчас:
- позже видео пойдут не только из файла, но и из живой камеры;
- GUI, сессия и пайплайны должны заранее знать, что такой тип источника вообще возможен.

Почему пока заглушка:
- на текущем этапе у нас нет стабильного живого потока для разработки;
- важнее не строить замок из интерфейсов, а сохранить чистую архитектурную точку расширения.
"""

from __future__ import annotations

from .base_frame_reader import BaseFrameSource


class RtspStreamFrameSource(BaseFrameSource):
    """Будущий источник кадров из RTSP-потока."""

    implementation_name = "rtsp_stream"
    is_ready = False

    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url

    def read(self):
        raise NotImplementedError("RTSP-источник пока не реализован в рабочем режиме.")

    def close(self) -> None:
        """Пока освобождать нечего."""
