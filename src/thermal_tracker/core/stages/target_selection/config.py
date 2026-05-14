from __future__ import annotations

from typing import TypeAlias

from .operations import (
    ContrastComponentTargetSelectorConfig,
    GrabCutTargetSelectorConfig,
)


TargetSelectionConfig: TypeAlias = (
    ContrastComponentTargetSelectorConfig
    | GrabCutTargetSelectorConfig
)


_TargetSelectionConfigClass: TypeAlias = (
    type[ContrastComponentTargetSelectorConfig]
    | type[GrabCutTargetSelectorConfig]
)


TARGET_SELECTION_CONFIG_CLASSES: dict[str, _TargetSelectionConfigClass] = {
    str(ContrastComponentTargetSelectorConfig.operation_type): (
        ContrastComponentTargetSelectorConfig
    ),
    str(GrabCutTargetSelectorConfig.operation_type): GrabCutTargetSelectorConfig,
}
