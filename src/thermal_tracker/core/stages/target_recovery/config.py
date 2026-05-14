from __future__ import annotations

from typing import TypeAlias

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
