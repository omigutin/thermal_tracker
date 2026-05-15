from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar


OperationConfigT = TypeVar("OperationConfigT")


@dataclass(frozen=True, slots=True)
class StageConfig(Generic[OperationConfigT]):
    """
        Общая конфигурация стадии, которая состоит из набора операций.

        Этот класс не описывает конкретную бизнес-логику стадии.
        Он хранит только общие для всех operation-stage параметры:
        включена ли стадия и какие операции должны выполняться внутри неё.

        Например, для candidate_filtering здесь будет лежать список фильтров,
        для frame_preprocessing — список операций обработки кадра,
        для target_tracking — список доступных трекеров.

        Конкретные параметры каждой операции хранятся не здесь, а в её собственном config-классе.
    """

    # Включает или отключает всю стадию.
    enabled: bool = False
    # Операции стадии в порядке выполнения.
    operations: tuple[OperationConfigT, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Проверить корректность конфигурации стадии."""
        if self.enabled and not self.operations:
            raise ValueError("Stage is enabled, but no operations are configured.")

    @property
    def enabled_operations(self) -> tuple[OperationConfigT, ...]:
        """Возвращает разрешённые операции, которые должны попасть в pipeline."""
        if not self.enabled:
            return ()
        return self.operations
