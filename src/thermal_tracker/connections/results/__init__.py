"""Result writer implementations."""

from .base_result_writer import BaseResultWriter
from .file_result_writer import FileResultWriter
from .gui_result_writer import GuiResultWriter
from .log_result_writer import LogResultWriter
from .shared_memory_result_writer import SharedMemoryResultWriter


def create_result_writer(config):
    writer = (getattr(config, "writer", "") or "").strip()
    if writer == "gui":
        return GuiResultWriter()
    if writer == "shared_memory":
        return SharedMemoryResultWriter()
    if writer == "file":
        return FileResultWriter()
    if writer in {"", "log", "null", "none"}:
        return LogResultWriter()
    raise ValueError(f"Unknown result writer: {writer!r}")


__all__ = [
    "BaseResultWriter",
    "FileResultWriter",
    "GuiResultWriter",
    "LogResultWriter",
    "SharedMemoryResultWriter",
    "create_result_writer",
]
