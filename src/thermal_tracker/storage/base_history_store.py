"""History store contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseHistoryStore(ABC):
    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def append(self, item: Any) -> None:
        """Store one history item."""

    def close(self) -> None:
        pass
