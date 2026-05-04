"""Shared memory command reader placeholder."""

from __future__ import annotations

from .base_command_reader import BaseCommandReader


class SharedMemoryCommandReader(BaseCommandReader):
    implementation_name = "shared_memory"
    is_ready = False

    def read(self):
        raise NotImplementedError("Shared memory command reader is not implemented yet.")
