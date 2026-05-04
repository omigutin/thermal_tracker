"""Command reader contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCommandReader(ABC):
    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def read(self) -> Any | None:
        """Read one operator/runtime command if available."""

    def close(self) -> None:
        pass
