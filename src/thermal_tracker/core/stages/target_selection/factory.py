from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import TargetSelectionConfig
from .operations import (
    BaseTargetSelector,
    ContrastComponentTargetSelector,
    ContrastComponentTargetSelectorConfig,
    GrabCutTargetSelector,
    GrabCutTargetSelectorConfig,
)


class TargetSelectionFactory:
    """Создаёт runtime-операции выбора цели из конфигураций."""

    @classmethod
    def build_many(cls, operations: Sequence[TargetSelectionConfig]) -> tuple[BaseTargetSelector, ...]:
        """Создать набор активных операций в исходном порядке."""
        result: list[BaseTargetSelector] = []

        for operation_config in operations:
            operation = cls.build(operation_config)
            if operation is not None:
                result.append(operation)

        return tuple(result)

    @classmethod
    def build(cls, operation_config: TargetSelectionConfig) -> BaseTargetSelector | None:
        """Создать одну runtime-операцию из конфигурации."""
        cls._validate_operation_config(operation_config)

        if not operation_config.enabled:
            return None

        return cls._build_runtime_operation(operation_config)

    @classmethod
    def _build_runtime_operation(cls, operation_config: TargetSelectionConfig) -> BaseTargetSelector:
        """Создать runtime-операцию по конкретному типу конфигурации."""
        if isinstance(operation_config, ContrastComponentTargetSelectorConfig):
            return ContrastComponentTargetSelector(config=operation_config)

        if isinstance(operation_config, GrabCutTargetSelectorConfig):
            return GrabCutTargetSelector(config=operation_config)

        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: object) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(operation_config, (ContrastComponentTargetSelectorConfig, GrabCutTargetSelectorConfig)):
            return

        TargetSelectionFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации выбора цели."""
        raise TypeError(f"Unsupported target selection config: {type(operation_config).__name__!r}.")
