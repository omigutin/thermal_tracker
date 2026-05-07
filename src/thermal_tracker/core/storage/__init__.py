"""Реализации хранилищ истории."""

from .base_history_store import BaseHistoryStore
from .file_history_store import FileHistoryStore
from .memory_history_store import MemoryHistoryStore
from .null_history_store import NullHistoryStore
from .sqlite_history_store import SqliteHistoryStore


def create_history_store(config):
    if not getattr(config, "enabled", False):
        return NullHistoryStore()
    store = (getattr(config, "store", "") or "").strip()
    if store == "memory":
        return MemoryHistoryStore()
    if store == "file":
        return FileHistoryStore()
    if store == "sqlite":
        return SqliteHistoryStore()
    return NullHistoryStore()


__all__ = [
    "BaseHistoryStore",
    "FileHistoryStore",
    "MemoryHistoryStore",
    "NullHistoryStore",
    "SqliteHistoryStore",
    "create_history_store",
]
