"""Минимальные тесты на TargetRecoveryConfig (stage-level параметры)."""

from __future__ import annotations

import pytest

from thermal_tracker.core.stages.config.stage_config import StageConfig
from thermal_tracker.core.stages.target_recovery.config import TargetRecoveryConfig


def _build_stage(enabled: bool = False) -> StageConfig[object]:
    """Собрать валидный StageConfig для подстановки в TargetRecoveryConfig."""
    return StageConfig(enabled=enabled, operations=())


def test_explicit_valid_config() -> None:
    """Явный корректный конфиг должен создаваться без ошибок."""
    config = TargetRecoveryConfig(
        enabled=True,
        stage=_build_stage(),
        min_lost_frames=4,
        confirm_frames=2,
        recovery_window_frames=20,
    )

    assert config.min_lost_frames == 4
    assert config.confirm_frames == 2
    assert config.recovery_window_frames == 20


def test_negative_min_lost_frames_raises() -> None:
    """Отрицательное min_lost_frames должно отлавливаться валидацией."""
    with pytest.raises(ValueError):
        TargetRecoveryConfig(
            stage=_build_stage(),
            min_lost_frames=-1,
        )


def test_zero_confirm_frames_raises() -> None:
    """confirm_frames должен быть строго положительным."""
    with pytest.raises(ValueError):
        TargetRecoveryConfig(
            stage=_build_stage(),
            confirm_frames=0,
        )


def test_zero_recovery_window_frames_raises() -> None:
    """recovery_window_frames должен быть строго положительным."""
    with pytest.raises(ValueError):
        TargetRecoveryConfig(
            stage=_build_stage(),
            recovery_window_frames=0,
        )


def test_operations_property_delegates_to_stage() -> None:
    """Свойство operations должно возвращать operations stage."""
    stage = _build_stage()
    config = TargetRecoveryConfig(stage=stage)

    assert config.operations == stage.operations
