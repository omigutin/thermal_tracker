"""Заготовка писателя результатов в GUI."""

from __future__ import annotations

from typing import Any

from .base_result_writer import BaseResultWriter


class GuiResultWriter(BaseResultWriter):
    implementation_name = "gui"
    is_ready = True

    def write(self, result: Any) -> None:
        self.last_result = result
