"""Пустое хранилище истории."""

from __future__ import annotations

from typing import Any

from .base_history_store import BaseHistoryStore


class NullHistoryStore(BaseHistoryStore):
    implementation_name = "null"
    is_ready = True

    def append(self, item: Any) -> None:
        pass
