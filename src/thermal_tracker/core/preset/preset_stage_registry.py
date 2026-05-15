from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from ..stages.candidate_filtering import (
    CANDIDATE_FILTER_CONFIG_CLASSES,
    CandidateFilterConfig,
)
from ..stages.candidate_formation import (
    CANDIDATE_FORMER_CONFIG_CLASSES,
    CandidateFormerConfig,
)
from ..stages.frame_preprocessing import (
    FRAME_PREPROCESSOR_CONFIG_CLASSES,
    FramePreprocessorConfig,
)
from ..stages.frame_stabilization import (
    FRAME_STABILIZER_CONFIG_CLASSES,
    FrameStabilizerConfig,
)
from ..stages.motion_localization import (
    MOTION_LOCALIZER_CONFIG_CLASSES,
    MotionLocalizationConfig,
)
from ..stages.target_recovery import (
    TARGET_RECOVERER_CONFIG_CLASSES,
    TargetRecovererConfig,
)
from ..stages.target_selection import (
    TARGET_SELECTION_CONFIG_CLASSES,
    TargetSelectionConfig,
)
from ..stages.target_tracking import (
    TARGET_TRACKER_CONFIG_CLASSES,
    TargetTrackerConfig,
)


StageOperationConfig: TypeAlias = (
    FramePreprocessorConfig
    | FrameStabilizerConfig
    | MotionLocalizationConfig
    | CandidateFormerConfig
    | CandidateFilterConfig
    | TargetSelectionConfig
    | TargetTrackerConfig
    | TargetRecovererConfig
)


StageOperationConfigClass: TypeAlias = type[StageOperationConfig]


@dataclass(frozen=True, slots=True)
class StageRegistryItem:
    """Описание поддерживаемого типа стадии пресета."""

    # Тип стадии, указанный в TOML: frame_preprocessing, target_tracking и т.д.
    stage_type: str
    # Человекочитаемое имя стадии для сообщений об ошибках.
    title: str
    # Словарь operation type -> config class.
    config_classes: dict[str, StageOperationConfigClass]


class StageRegistry:
    """Хранит соответствие между типом стадии и её operation config-классами."""

    # Зарегистрированные типы стадий, которые можно использовать в пресете.
    _ITEMS: dict[str, StageRegistryItem] = {
        "frame_preprocessing": StageRegistryItem(
            stage_type="frame_preprocessing",
            title="Frame preprocessing",
            config_classes=FRAME_PREPROCESSOR_CONFIG_CLASSES,
        ),
        "frame_stabilization": StageRegistryItem(
            stage_type="frame_stabilization",
            title="Frame stabilization",
            config_classes=FRAME_STABILIZER_CONFIG_CLASSES,
        ),
        "motion_localization": StageRegistryItem(
            stage_type="motion_localization",
            title="Motion localization",
            config_classes=MOTION_LOCALIZER_CONFIG_CLASSES,
        ),
        "candidate_formation": StageRegistryItem(
            stage_type="candidate_formation",
            title="Candidate formation",
            config_classes=CANDIDATE_FORMER_CONFIG_CLASSES,
        ),
        "candidate_filtering": StageRegistryItem(
            stage_type="candidate_filtering",
            title="Candidate filtering",
            config_classes=CANDIDATE_FILTER_CONFIG_CLASSES,
        ),
        "target_selection": StageRegistryItem(
            stage_type="target_selection",
            title="Target selection",
            config_classes=TARGET_SELECTION_CONFIG_CLASSES,
        ),
        "target_tracking": StageRegistryItem(
            stage_type="target_tracking",
            title="Target tracking",
            config_classes=TARGET_TRACKER_CONFIG_CLASSES,
        ),
        "target_recovery": StageRegistryItem(
            stage_type="target_recovery",
            title="Target recovery",
            config_classes=TARGET_RECOVERER_CONFIG_CLASSES,
        ),
    }

    @classmethod
    def get(cls, stage_type: str) -> StageRegistryItem:
        """Вернуть описание стадии по её типу."""
        normalized_type = stage_type.strip()

        if not normalized_type:
            raise ValueError("stage type must not be empty.")

        item = cls._ITEMS.get(normalized_type)

        if item is None:
            raise ValueError(
                f"Unsupported preset stage type: {normalized_type!r}. "
                f"Available types: {tuple(sorted(cls._ITEMS))}."
            )

        return item

    @classmethod
    def has(cls, stage_type: str) -> bool:
        """Проверить, зарегистрирован ли тип стадии."""
        return stage_type.strip() in cls._ITEMS

    @classmethod
    def available_stage_types(cls) -> tuple[str, ...]:
        """Вернуть список доступных типов стадий."""
        return tuple(sorted(cls._ITEMS))
