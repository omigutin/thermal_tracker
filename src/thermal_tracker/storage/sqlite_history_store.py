"""SQLite history store placeholder."""

from __future__ import annotations

from typing import Any

from .base_history_store import BaseHistoryStore


class SqliteHistoryStore(BaseHistoryStore):
    implementation_name = "sqlite"
    is_ready = False

    def append(self, item: Any) -> None:
        raise NotImplementedError("SQLite history store is not implemented yet.")
