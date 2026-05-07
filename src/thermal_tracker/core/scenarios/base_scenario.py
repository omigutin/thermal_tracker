"""Общий протокол сценария."""

from __future__ import annotations

from typing import Protocol


class BaseScenario(Protocol):
    @property
    def preset_name(self) -> str:
        """Короткое имя пресета, используемого сценарием."""

    def process_next_raw_frame(self, *args, **kwargs):
        """Обрабатывает один сырой кадр."""
