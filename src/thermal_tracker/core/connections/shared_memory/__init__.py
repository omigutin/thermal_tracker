"""Общий слой Shared Memory для кадров, команд и результатов.

Этот пакет не знает про GUI, Web и конкретные сценарии. Его задача уже:
дать процессам один понятный контракт обмена данными внутри одной машины.
"""

from .frame_buffer import SharedMemoryFrameBuffer, SharedMemoryFrame
from .json_buffer import SharedMemoryJsonBuffer
from .protocol import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_CHANNELS,
    DEFAULT_FRAME_FORMAT,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
    FRAME_FORMAT_RAW_Y8,
)

__all__ = [
    "DEFAULT_CAMERA_ID",
    "DEFAULT_FRAME_CHANNELS",
    "DEFAULT_FRAME_FORMAT",
    "DEFAULT_FRAME_HEIGHT",
    "DEFAULT_FRAME_WIDTH",
    "DEFAULT_SHARED_MEMORY_PREFIX",
    "FRAME_FORMAT_RAW_Y8",
    "SharedMemoryFrame",
    "SharedMemoryFrameBuffer",
    "SharedMemoryJsonBuffer",
]
