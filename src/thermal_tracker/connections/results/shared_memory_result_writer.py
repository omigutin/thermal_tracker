"""Shared memory result writer placeholder."""

from __future__ import annotations

from typing import Any

from .base_result_writer import BaseResultWriter


class SharedMemoryResultWriter(BaseResultWriter):
    implementation_name = "shared_memory"
    is_ready = False

    def write(self, result: Any) -> None:
        raise NotImplementedError("Shared memory result writer is not implemented yet.")
