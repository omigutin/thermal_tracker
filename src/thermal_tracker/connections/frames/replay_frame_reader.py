"""Источник кадров из заранее переданного списка.

Это удобный инженерный режим:
- можно подать уже загруженные кадры;
- легко воспроизводить один и тот же кусок сцены;
- удобно для тестов и мини-бенчмарков.
"""

from __future__ import annotations

import numpy as np

from .base_frame_reader import BaseFrameSource


class ReplayFrameSource(BaseFrameSource):
    """Воспроизводит кадры из памяти один за другим."""

    implementation_name = "replay"
    is_ready = True

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = [frame.copy() for frame in frames]
        self._index = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def close(self) -> None:
        self._frames = []
