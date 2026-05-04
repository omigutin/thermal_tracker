"""Реализации читателей команд."""

from .base_command_reader import BaseCommandReader
from .gui_command_reader import GuiCommandReader
from .null_command_reader import NullCommandReader
from .shared_memory_command_reader import SharedMemoryCommandReader


def create_command_reader(config):
    reader = (getattr(config, "reader", "") or "").strip()
    if reader == "gui":
        return GuiCommandReader()
    if reader == "shared_memory":
        return SharedMemoryCommandReader()
    if reader in {"", "null", "none"}:
        return NullCommandReader()
    raise ValueError(f"Unknown command reader: {reader!r}")


__all__ = [
    "BaseCommandReader",
    "GuiCommandReader",
    "NullCommandReader",
    "SharedMemoryCommandReader",
    "create_command_reader",
]
