"""Common scenario protocol."""

from __future__ import annotations

from typing import Protocol


class BaseScenario(Protocol):
    @property
    def preset_name(self) -> str:
        """Short preset name used by the scenario."""

    def process_next_raw_frame(self, *args, **kwargs):
        """Process one raw frame."""
