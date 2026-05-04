"""Контракт хранилища истории."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseHistoryStore(ABC):
    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def append(self, item: Any) -> None:
        """Сохраняет одну запись истории."""

    def close(self) -> None:
        pass
