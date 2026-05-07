"""Читатель кадров из Shared Memory."""

from __future__ import annotations

import numpy as np

from ..shared_memory import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_CHANNELS,
    DEFAULT_FRAME_FORMAT,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
    SharedMemoryFrame,
    SharedMemoryFrameBuffer,
)
from .base_frame_reader import BaseFrameSource


class SharedMemoryFrameReader(BaseFrameSource):
    implementation_name = "shared_memory"
    is_ready = True

    def __init__(
        self,
        camera_count: int = 1,
        *,
        prefix: str = DEFAULT_SHARED_MEMORY_PREFIX,
        camera_id: int = DEFAULT_CAMERA_ID,
        width: int = DEFAULT_FRAME_WIDTH,
        height: int = DEFAULT_FRAME_HEIGHT,
        channels: int = DEFAULT_FRAME_CHANNELS,
        frame_format: str = DEFAULT_FRAME_FORMAT,
    ) -> None:
        self.camera_count = int(camera_count)
        self.prefix = prefix
        self.camera_id = int(camera_id)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.frame_format = frame_format
        self.last_frame: SharedMemoryFrame | None = None
        self._last_sequence = 0
        self._buffer: SharedMemoryFrameBuffer | None = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Читает последний новый кадр из Shared Memory."""

        buffer = self._ensure_buffer()
        if buffer is None:
            return False, None

        frame = buffer.read_latest(after_sequence=self._last_sequence)
        if frame is None:
            return False, None

        self.last_frame = frame
        self._last_sequence = frame.sequence
        return True, frame.image

    def close(self) -> None:
        """Закрывает подключение к Shared Memory."""

        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def _ensure_buffer(self) -> SharedMemoryFrameBuffer | None:
        """Лениво подключается к кадровому буферу, когда он появится."""

        if self._buffer is not None:
            return self._buffer
        self._buffer = SharedMemoryFrameBuffer.try_open(
            prefix=self.prefix,
            camera_id=self.camera_id,
            width=self.width,
            height=self.height,
            channels=self.channels,
            frame_format=self.frame_format,
        )
        return self._buffer
