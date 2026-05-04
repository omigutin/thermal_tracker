"""Result writer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseResultWriter(ABC):
    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def write(self, result: Any) -> None:
        """Write one processing result."""

    def close(self) -> None:
        pass
