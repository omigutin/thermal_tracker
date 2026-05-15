from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from thermal_tracker.core.stages.config.stage_config import StageConfig
from .operations import (
    IrstContrastTargetRecovererConfig,
    LocalTemplateTargetRecovererConfig,
)


TargetRecovererConfig: TypeAlias = (
    LocalTemplateTargetRecovererConfig
    | IrstContrastTargetRecovererConfig
)


_TargetRecovererConfigClass: TypeAlias = (
    type[LocalTemplateTargetRecovererConfig]
    | type[IrstContrastTargetRecovererConfig]
)


TARGET_RECOVERER_CONFIG_CLASSES: dict[str, _TargetRecovererConfigClass] = {
    str(LocalTemplateTargetRecovererConfig.operation_type): LocalTemplateTargetRecovererConfig,
    str(IrstContrastTargetRecovererConfig.operation_type): IrstContrastTargetRecovererConfig,
}


@dataclass(frozen=True, slots=True)
class TargetRecoveryConfig:
    """Настройки стадии восстановления потерянной цели."""

    # Включает или отключает всю стадию восстановления цели.
    enabled: bool = True
    # Конфигурация операций восстановления цели.
    stage: StageConfig[TargetRecovererConfig] = field(
        default_factory=lambda: StageConfig[TargetRecovererConfig](
            enabled=True,
            operations=(),
        )
    )
    # После скольких подряд потерянных кадров pipeline запускает recovery.
    min_lost_frames: int = 5
    # Сколько подряд подтверждений нужно для возврата из RECOVERING в TRACKING.
    confirm_frames: int = 3
    # Максимальное окно восстановления перед окончательной потерей цели.
    recovery_window_frames: int = 30

    def __post_init__(self) -> None:
        """Проверить корректность настроек стадии восстановления цели."""
        self._validate_non_negative_int(self.min_lost_frames, "min_lost_frames")
        self._validate_positive_int(self.confirm_frames, "confirm_frames")
        self._validate_positive_int(
            self.recovery_window_frames,
            "recovery_window_frames",
        )

    @property
    def operations(self) -> tuple[TargetRecovererConfig, ...]:
        """Вернуть конфигурации операций восстановления цели."""
        return self.stage.operations

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

    @staticmethod
    def _validate_non_negative_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")
