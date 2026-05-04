"""Контракт читателя команд."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCommandReader(ABC):
    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def read(self) -> Any | None:
        """Читает одну команду оператора или runtime, если она доступна."""

    def close(self) -> None:
        pass
