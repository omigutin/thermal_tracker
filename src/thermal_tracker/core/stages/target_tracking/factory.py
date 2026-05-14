from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import TARGET_TRACKER_CONFIG_CLASSES, TargetTrackerConfig
from .operations import (
    BaseTargetTracker,
    CsrtTargetTracker,
    CsrtTargetTrackerConfig,
    IrstContrastTargetTracker,
    IrstContrastTargetTrackerConfig,
    TemplatePointTargetTracker,
    TemplatePointTargetTrackerConfig,
    YoloTargetTracker,
    YoloTargetTrackerConfig,
)


class TargetTrackerFactory:
    """Создаёт runtime-трекеры цели из конфигураций."""

    @classmethod
    def build_many(
        cls,
        operations: Sequence[TargetTrackerConfig],
    ) -> tuple[BaseTargetTracker, ...]:
        """Создать набор активных трекеров в исходном порядке."""
        result: list[BaseTargetTracker] = []

        for operation_config in operations:
            tracker = cls.build(operation_config)
            if tracker is not None:
                result.append(tracker)

        return tuple(result)

    @classmethod
    def build(
        cls,
        operation_config: TargetTrackerConfig,
    ) -> BaseTargetTracker | None:
        """Создать один runtime-трекер из конфигурации."""
        cls._validate_operation_config(operation_config)

        if not operation_config.enabled:
            return None

        return cls._build_runtime_tracker(operation_config)

    @classmethod
    def _build_runtime_tracker(
        cls,
        operation_config: TargetTrackerConfig,
    ) -> BaseTargetTracker:
        """Создать runtime-трекер по конкретному типу конфигурации."""
        if isinstance(operation_config, TemplatePointTargetTrackerConfig):
            return TemplatePointTargetTracker(config=operation_config)

        if isinstance(operation_config, CsrtTargetTrackerConfig):
            return CsrtTargetTracker(config=operation_config)

        if isinstance(operation_config, IrstContrastTargetTrackerConfig):
            return IrstContrastTargetTracker(config=operation_config)

        if isinstance(operation_config, YoloTargetTrackerConfig):
            return YoloTargetTracker(config=operation_config)

        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: object) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(
            operation_config,
            tuple(TARGET_TRACKER_CONFIG_CLASSES.values()),
        ):
            return

        TargetTrackerFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации трекера."""
        raise TypeError(
            f"Unsupported target tracker config: "
            f"{type(operation_config).__name__!r}."
        )
