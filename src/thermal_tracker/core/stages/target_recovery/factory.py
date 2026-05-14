from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import TARGET_RECOVERER_CONFIG_CLASSES, TargetRecovererConfig
from .operations import (
    BaseTargetRecoverer,
    IrstContrastTargetRecoverer,
    IrstContrastTargetRecovererConfig,
    LocalTemplateTargetRecoverer,
    LocalTemplateTargetRecovererConfig,
)


class TargetRecovererFactory:
    """Создаёт runtime-операции восстановления цели из конфигураций."""

    @classmethod
    def build_many(
        cls,
        operations: Sequence[TargetRecovererConfig],
    ) -> tuple[BaseTargetRecoverer, ...]:
        """Создать набор активных recoverer-ов в исходном порядке."""
        result: list[BaseTargetRecoverer] = []

        for operation_config in operations:
            recoverer = cls.build(operation_config)
            if recoverer is not None:
                result.append(recoverer)

        return tuple(result)

    @classmethod
    def build(
        cls,
        operation_config: TargetRecovererConfig,
    ) -> BaseTargetRecoverer | None:
        """Создать один runtime-recoverer из конфигурации."""
        cls._validate_operation_config(operation_config)

        if not operation_config.enabled:
            return None

        return cls._build_runtime_recoverer(operation_config)

    @classmethod
    def _build_runtime_recoverer(
        cls,
        operation_config: TargetRecovererConfig,
    ) -> BaseTargetRecoverer:
        """Создать runtime-recoverer по конкретному типу конфигурации."""
        if isinstance(operation_config, LocalTemplateTargetRecovererConfig):
            return LocalTemplateTargetRecoverer(config=operation_config)

        if isinstance(operation_config, IrstContrastTargetRecovererConfig):
            return IrstContrastTargetRecoverer(config=operation_config)

        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: object) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(
            operation_config,
            tuple(TARGET_RECOVERER_CONFIG_CLASSES.values()),
        ):
            return

        TargetRecovererFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации recoverer-а."""
        raise TypeError(
            f"Unsupported target recoverer config: "
            f"{type(operation_config).__name__!r}."
        )
