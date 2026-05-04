"""Пустой читатель команд."""

from __future__ import annotations

from .base_command_reader import BaseCommandReader


class NullCommandReader(BaseCommandReader):
    implementation_name = "null"
    is_ready = True

    def read(self):
        return None
