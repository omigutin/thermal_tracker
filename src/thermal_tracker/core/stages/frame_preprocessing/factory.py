from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .config import FRAME_PREPROCESSOR_CONFIG_CLASSES, FramePreprocessorConfig
from .operations import (
    BaseFramePreprocessor,
    BilateralBlurFramePreprocessor,
    BilateralBlurFramePreprocessorConfig,
    ClaheContrastFramePreprocessor,
    ClaheContrastFramePreprocessorConfig,
    GaussianBlurFramePreprocessor,
    GaussianBlurFramePreprocessorConfig,
    GradientFramePreprocessor,
    GradientFramePreprocessorConfig,
    MedianBlurFramePreprocessor,
    MedianBlurFramePreprocessorConfig,
    MinMaxNormalizeFramePreprocessor,
    MinMaxNormalizeFramePreprocessorConfig,
    PercentileNormalizeFramePreprocessor,
    PercentileNormalizeFramePreprocessorConfig,
    ResizeFramePreprocessor,
    ResizeFramePreprocessorConfig,
    SharpnessMetricFramePreprocessor,
    SharpnessMetricFramePreprocessorConfig,
)


class FramePreprocessorFactory:
    """Создаёт runtime-операции предобработки кадра из конфигураций."""

    @classmethod
    def build_many(
        cls,
        operations: Sequence[FramePreprocessorConfig],
    ) -> tuple[BaseFramePreprocessor, ...]:
        """Создать набор активных операций в исходном порядке."""
        result: list[BaseFramePreprocessor] = []

        for operation_config in operations:
            preprocessor = cls.build(operation_config)
            if preprocessor is not None:
                result.append(preprocessor)

        return tuple(result)

    @classmethod
    def build(
        cls,
        operation_config: FramePreprocessorConfig,
    ) -> BaseFramePreprocessor | None:
        """Создать одну runtime-операцию из конфигурации."""
        cls._validate_operation_config(operation_config)

        if not operation_config.enabled:
            return None

        return cls._build_runtime_preprocessor(operation_config)

    @classmethod
    def _build_runtime_preprocessor(
        cls,
        operation_config: FramePreprocessorConfig,
    ) -> BaseFramePreprocessor:
        """Создать runtime-операцию по конкретному типу конфигурации."""
        if isinstance(operation_config, ResizeFramePreprocessorConfig):
            return ResizeFramePreprocessor(config=operation_config)

        if isinstance(operation_config, GaussianBlurFramePreprocessorConfig):
            return GaussianBlurFramePreprocessor(config=operation_config)

        if isinstance(operation_config, MedianBlurFramePreprocessorConfig):
            return MedianBlurFramePreprocessor(config=operation_config)

        if isinstance(operation_config, BilateralBlurFramePreprocessorConfig):
            return BilateralBlurFramePreprocessor(config=operation_config)

        if isinstance(operation_config, MinMaxNormalizeFramePreprocessorConfig):
            return MinMaxNormalizeFramePreprocessor(config=operation_config)

        if isinstance(operation_config, PercentileNormalizeFramePreprocessorConfig):
            return PercentileNormalizeFramePreprocessor(config=operation_config)

        if isinstance(operation_config, ClaheContrastFramePreprocessorConfig):
            return ClaheContrastFramePreprocessor(config=operation_config)

        if isinstance(operation_config, GradientFramePreprocessorConfig):
            return GradientFramePreprocessor(config=operation_config)

        if isinstance(operation_config, SharpnessMetricFramePreprocessorConfig):
            return SharpnessMetricFramePreprocessor(config=operation_config)

        cls._raise_invalid_config(operation_config)

    @staticmethod
    def _validate_operation_config(operation_config: object) -> None:
        """Проверить, что фабрика поддерживает переданную конфигурацию."""
        if isinstance(
            operation_config,
            tuple(FRAME_PREPROCESSOR_CONFIG_CLASSES.values()),
        ):
            return

        FramePreprocessorFactory._raise_invalid_config(operation_config)

    @staticmethod
    def _raise_invalid_config(operation_config: object) -> NoReturn:
        """Выбросить ошибку неподдерживаемой конфигурации препроцессора."""
        raise TypeError(
            f"Unsupported frame preprocessor config: "
            f"{type(operation_config).__name__!r}."
        )
