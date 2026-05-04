"""Shared memory frame reader placeholder."""

from __future__ import annotations

import numpy as np

from .base_frame_reader import BaseFrameSource


class SharedMemoryFrameReader(BaseFrameSource):
    implementation_name = "shared_memory"
    is_ready = False

    def __init__(self, camera_count: int = 1) -> None:
        self.camera_count = camera_count

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError("Shared memory frame reader is not implemented yet.")

    def close(self) -> None:
        pass
