from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import MotionLocalizationConfig
from .operations.base_motion_localizer import BaseMotionLocalizer
from .operations.frame_difference_motion_localizer import (
    FrameDifferenceMotionLocalizer,
    FrameDifferenceMotionLocalizerConfig
)
from .operations.knn_motion_localizer import (
    KnnMotionLocalizer,
    KnnMotionLocalizerConfig
)
from .operations.mog2_motion_localizer import (
    Mog2MotionLocalizer,
    Mog2MotionLocalizerConfig
)
from .operations.running_average_motion_localizer import (
    RunningAverageMotionLocalizer,
    RunningAverageMotionLocalizerConfig
)


class MotionLocalizationFactory:
    """Создаёт runtime-операции локализации движения из конфигураций."""

    @classmethod
    def build_many(cls, operations: Sequence[MotionLocalizationConfig]) -> tuple[BaseMotionLocalizer, ...]:
        """Создать набор активных операций в исходном порядке."""
        result: list[BaseMotionLocalizer] = []

        for operation_config in operations:
            operation = cls.build(operation_config)
            if operation is not None:
                result.append(operation)

        return tuple(result)

    @classmethod
    def build(cls, operation_config: MotionLocalizationConfig) -> BaseMotionLocalizer | None:
        """Создать одну runtime-операцию из конфигурации."""
        cls._validate_operation_config(operation_config)

        if not operation_config.enabled:
            return None

        return cls._build_runtime_operation(operation_config)

    @classmethod
    def _build_runtime_operation(cls, operation_config: MotionLocalizationConfig) -> BaseMotionLocalizer:
        """Создать runtime-операцию по конкретному типу конфигурации."""
        if isinstance(operation_config, FrameDifferenceMotionLocalizerConfig):
            return FrameDifferenceMotionLocalizer(config=operation_config)

        if isinstance(operation_config, KnnMotionLocalizerConfig):
            return KnnMotionLocalizer(config=operation_config)

        if isinstance(operation_config, Mog2MotionLocalizerConfig):
            return Mog2MotionLocalizer(config=operation_config)

        if isinstance(operation_config, RunningAverageMotionLocalizerConfig):
            return RunningAverageMotionLocalizer(config=operation_config)

        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: object) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(
            operation_config,
            (
                    FrameDifferenceMotionLocalizerConfig,
                    KnnMotionLocalizerConfig,
                    Mog2MotionLocalizerConfig,
                    RunningAverageMotionLocalizerConfig,
            ),
        ):
            return
        MotionLocalizationFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации локализации движения."""
        raise TypeError(f"Unsupported motion localization config: {type(operation_config).__name__!r}.")
