"""File history store placeholder."""

from __future__ import annotations

from typing import Any

from .base_history_store import BaseHistoryStore


class FileHistoryStore(BaseHistoryStore):
    implementation_name = "file"
    is_ready = False

    def append(self, item: Any) -> None:
        raise NotImplementedError("File history store is not implemented yet.")
