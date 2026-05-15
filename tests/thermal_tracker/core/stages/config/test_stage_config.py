"""Минимальные тесты на инвариант StageConfig."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from thermal_tracker.core.stages.config.stage_config import StageConfig


@dataclass(frozen=True, slots=True)
class _DummyConfig:
    """Заглушка под operation config для проверки StageConfig."""
    name: str = "dummy"


def test_disabled_with_no_operations_is_valid() -> None:
    """StageConfig(enabled=False, operations=()) должен создаваться без ошибок."""
    config: StageConfig[_DummyConfig] = StageConfig(enabled=False, operations=())

    assert config.enabled is False
    assert config.operations == ()


def test_enabled_with_operations_is_valid() -> None:
    """Включённая стадия с операциями должна создаваться без ошибок."""
    operations = (_DummyConfig(name="a"), _DummyConfig(name="b"))
    config: StageConfig[_DummyConfig] = StageConfig(enabled=True, operations=operations)

    assert config.enabled is True
    assert config.operations == operations


def test_enabled_without_operations_raises() -> None:
    """Включённая пустая стадия должна падать на этапе валидации."""
    with pytest.raises(ValueError):
        StageConfig(enabled=True, operations=())


def test_enabled_operations_empty_when_disabled() -> None:
    """enabled_operations должен быть пуст для выключенной стадии."""
    operations = (_DummyConfig(name="a"),)
    config: StageConfig[_DummyConfig] = StageConfig(enabled=False, operations=operations)

    assert config.enabled_operations == ()


def test_enabled_operations_returns_all_when_enabled() -> None:
    """enabled_operations должен вернуть все операции, если стадия включена."""
    operations = (_DummyConfig(name="a"), _DummyConfig(name="b"))
    config: StageConfig[_DummyConfig] = StageConfig(enabled=True, operations=operations)

    assert config.enabled_operations == operations
