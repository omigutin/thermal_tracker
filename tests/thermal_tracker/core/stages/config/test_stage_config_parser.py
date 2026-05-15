"""Минимальные тесты на парсинг секции стадии в StageConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from thermal_tracker.core.stages.config.stage_config_parser import StageConfigParser


@dataclass(frozen=True, slots=True)
class _OpConfigA:
    """Заглушка под operation config с типом 'a'."""

    operation_type: ClassVar[str] = "a"
    enabled: bool = True
    value: int = 0

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "_OpConfigA":
        """Собрать конфиг по сырым значениям."""
        return cls(
            enabled=bool(values.get("enabled", True)),
            value=int(values.get("value", 0)),
        )


@dataclass(frozen=True, slots=True)
class _OpConfigB:
    """Заглушка под operation config с типом 'b'."""

    operation_type: ClassVar[str] = "b"
    enabled: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "_OpConfigB":
        """Собрать конфиг по сырым значениям."""
        return cls(enabled=bool(values.get("enabled", True)))


_CONFIG_CLASSES = {"a": _OpConfigA, "b": _OpConfigB}


def test_empty_section_returns_disabled_stage() -> None:
    """Пустая секция = выключенная стадия без операций."""
    result = StageConfigParser.parse(
        section={}, stage_name="dummy", config_classes=_CONFIG_CLASSES,
    )

    assert result.enabled is False
    assert result.operations == ()


def test_no_enabled_key_defaults_to_enabled_true() -> None:
    """Если ключ enabled не задан, стадия трактуется как включённая."""
    section: dict[str, object] = {
        "operations": [{"type": "a", "value": 7}],
    }
    result = StageConfigParser.parse(
        section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
    )

    assert result.enabled is True
    assert len(result.operations) == 1
    assert isinstance(result.operations[0], _OpConfigA)
    assert result.operations[0].value == 7


def test_enabled_with_operations_parses_all_types() -> None:
    """Парсер должен пройти все элементы operations и сопоставить их по type."""
    section: dict[str, object] = {
        "enabled": True,
        "operations": [
            {"type": "a", "value": 1},
            {"type": "b"},
        ],
    }
    result = StageConfigParser.parse(
        section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
    )

    assert result.enabled is True
    assert isinstance(result.operations[0], _OpConfigA)
    assert isinstance(result.operations[1], _OpConfigB)


def test_unknown_operation_type_raises() -> None:
    """Неизвестный тип операции должен приводить к RuntimeError."""
    section: dict[str, object] = {"operations": [{"type": "unknown"}]}

    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
        )


def test_missing_operation_type_raises() -> None:
    """Operation без поля type должна падать на парсинге."""
    section: dict[str, object] = {"operations": [{"value": 1}]}

    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
        )


def test_legacy_filters_key_raises() -> None:
    """Старый ключ filters должен явно сообщаться как legacy."""
    section: dict[str, object] = {"filters": []}

    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
        )


def test_legacy_methods_key_raises() -> None:
    """Старый ключ methods должен явно сообщаться как legacy."""
    section: dict[str, object] = {"methods": []}

    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
        )


def test_operations_must_be_array() -> None:
    """Operations должен быть массивом, а не словарём или строкой."""
    section: dict[str, object] = {"operations": {"type": "a"}}

    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
        )


def test_no_registered_classes_raises() -> None:
    """Стадия без зарегистрированных config-классов считается ошибкой."""
    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section={"operations": []}, stage_name="empty", config_classes={},
        )


def test_enabled_must_be_bool() -> None:
    """Значение enabled должно быть строго bool."""
    section: dict[str, object] = {"enabled": "yes", "operations": [{"type": "a"}]}

    with pytest.raises(RuntimeError):
        StageConfigParser.parse(
            section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
        )


def test_disabled_with_empty_operations_is_valid() -> None:
    """Стадию можно явно выключить и не задавать operations."""
    section: dict[str, object] = {"enabled": False}
    result = StageConfigParser.parse(
        section=section, stage_name="dummy", config_classes=_CONFIG_CLASSES,
    )

    assert result.enabled is False
    assert result.operations == ()
