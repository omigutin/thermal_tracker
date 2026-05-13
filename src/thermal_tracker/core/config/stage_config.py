from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar


OperationConfigT = TypeVar("OperationConfigT")


@dataclass(frozen=True, slots=True)
class StageConfig(Generic[OperationConfigT]):
    """Конфигурация стадии с упорядоченным набором операций."""

    # Включает или отключает всю стадию.
    enabled: bool = False
    # Операции стадии в порядке выполнения.
    operations: tuple[OperationConfigT, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Проверить корректность конфигурации стадии."""
        if self.enabled and not self.operations:
            raise ValueError("Stage is enabled, but no operations are configured.")

    @property
    def runtime_operations(self) -> tuple[OperationConfigT, ...]:
        """Вернуть операции, которые должны попасть в runtime pipeline."""
        if not self.enabled:
            return ()
        return self.operations
