from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import CandidateFormerConfig
from .operations import (
    BaseCandidateFormer,
    ConnectedComponentsCandidateFormer,
    ConnectedComponentsCandidateFormerConfig,
    ContourCandidateFormer,
    ContourCandidateFormerConfig,
)


class CandidateFormationFactory:
    """Создаёт runtime-операции формирования кандидатов из конфигураций."""

    @classmethod
    def build_many(cls, operations: Sequence[CandidateFormerConfig]) -> tuple[BaseCandidateFormer, ...]:
        """Создать набор активных операций в исходном порядке."""
        result: list[BaseCandidateFormer] = []

        for operation_config in operations:
            operation = cls.build(operation_config)
            if operation is not None:
                result.append(operation)

        return tuple(result)

    @classmethod
    def build(cls, operation_config: CandidateFormerConfig) -> BaseCandidateFormer | None:
        """Создать одну runtime-операцию из конфигурации."""
        cls._validate_operation_config(operation_config)

        if not operation_config.enabled:
            return None

        return cls._build_runtime_operation(operation_config)

    @classmethod
    def _build_runtime_operation(cls, operation_config: CandidateFormerConfig) -> BaseCandidateFormer:
        """Создать runtime-операцию по конкретному типу конфигурации."""
        if isinstance(operation_config, ConnectedComponentsCandidateFormerConfig):
            return ConnectedComponentsCandidateFormer(config=operation_config)

        if isinstance(operation_config, ContourCandidateFormerConfig):
            return ContourCandidateFormer(config=operation_config)

        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: object) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(
            operation_config,
            (
                ConnectedComponentsCandidateFormerConfig,
                ContourCandidateFormerConfig,
            ),
        ):
            return

        CandidateFormationFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации формирования кандидатов."""
        raise TypeError(f"Unsupported candidate formation config: {type(operation_config).__name__!r}.")
