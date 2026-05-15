"""Минимальные тесты на чтение типизированных полей через PresetFieldReader."""

from __future__ import annotations

import pytest

from thermal_tracker.core.preset.preset_field_reader import PresetFieldReader


def test_pop_int_reads_explicit_value() -> None:
    """Целое число должно попасть в target dict при явном указании."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"radius": 12})

    reader.pop_int_to(target, "radius")

    assert target == {"radius": 12}
    assert reader.values == {}


def test_pop_int_rejects_bool() -> None:
    """Bool не должен считаться целым числом."""
    reader = PresetFieldReader(owner="test", values={"radius": True})

    with pytest.raises(RuntimeError):
        reader.pop_int_to({}, "radius")


def test_pop_int_rejects_float() -> None:
    """Float не должен подменять собой int."""
    reader = PresetFieldReader(owner="test", values={"radius": 1.5})

    with pytest.raises(RuntimeError):
        reader.pop_int_to({}, "radius")


def test_pop_float_accepts_int() -> None:
    """Целое число должно приводиться к float."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"sigma": 2})

    reader.pop_float_to(target, "sigma")

    assert target == {"sigma": 2.0}
    assert isinstance(target["sigma"], float)


def test_pop_bool_reads_value() -> None:
    """Bool должен попасть в target dict."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"enabled": False})

    reader.pop_bool_to(target, "enabled")

    assert target == {"enabled": False}


def test_pop_str_reads_value() -> None:
    """Строка должна попасть в target dict."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"mode": "fast"})

    reader.pop_str_to(target, "mode")

    assert target == {"mode": "fast"}


def test_pop_int_tuple_reads_sequence() -> None:
    """Последовательность целых чисел должна стать tuple[int, ...]."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"kernel": [3, 5, 7]})

    reader.pop_int_tuple_to(target, "kernel")

    assert target == {"kernel": (3, 5, 7)}


def test_pop_float_tuple_reads_sequence() -> None:
    """Последовательность чисел должна стать tuple[float, ...]."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"scales": [0.8, 1, 1.2]})

    reader.pop_float_tuple_to(target, "scales")

    assert target == {"scales": (0.8, 1.0, 1.2)}


def test_pop_str_tuple_reads_sequence() -> None:
    """Последовательность строк должна стать tuple[str, ...]."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={"names": ["a", "b"]})

    reader.pop_str_tuple_to(target, "names")

    assert target == {"names": ("a", "b")}


def test_ensure_empty_does_not_raise_when_clean() -> None:
    """ensure_empty не должен бросать, если все поля прочитаны."""
    reader = PresetFieldReader(owner="test", values={"x": 1})
    reader.pop_int_to({}, "x")
    reader.ensure_empty()


def test_ensure_empty_raises_on_unknown_fields() -> None:
    """ensure_empty должен сообщать о незаявленных параметрах."""
    reader = PresetFieldReader(owner="test", values={"x": 1, "y": 2})

    with pytest.raises(RuntimeError):
        reader.ensure_empty()


def test_constructor_does_not_mutate_input_dict() -> None:
    """Конструктор не должен изменять внешний словарь."""
    source: dict[str, object] = {"a": 1}
    reader = PresetFieldReader(owner="test", values=source)
    reader.pop_int_to({}, "a")

    assert source == {"a": 1}
    assert reader.values == {}


def test_missing_key_is_silent() -> None:
    """Отсутствующее поле не должно добавляться в target dict."""
    target: dict[str, object] = {}
    reader = PresetFieldReader(owner="test", values={})

    reader.pop_int_to(target, "absent")
    reader.pop_float_to(target, "absent")
    reader.pop_bool_to(target, "absent")
    reader.pop_str_to(target, "absent")

    assert target == {}
