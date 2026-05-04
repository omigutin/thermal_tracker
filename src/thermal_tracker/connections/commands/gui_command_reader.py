"""GUI command reader placeholder."""

from __future__ import annotations

from .base_command_reader import BaseCommandReader


class GuiCommandReader(BaseCommandReader):
    implementation_name = "gui"
    is_ready = True

    def read(self):
        return None
