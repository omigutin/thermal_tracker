"""In-memory history store."""

from __future__ import annotations

from typing import Any

from .base_history_store import BaseHistoryStore


class MemoryHistoryStore(BaseHistoryStore):
    implementation_name = "memory"
    is_ready = True

    def __init__(self) -> None:
        self.items: list[Any] = []

    def append(self, item: Any) -> None:
        self.items.append(item)
