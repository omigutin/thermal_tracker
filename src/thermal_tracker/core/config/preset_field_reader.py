from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


_MISSING: Final[object] = object()


@dataclass(slots=True)
class PresetFieldReader:
    """Читает типизированные поля из сырого описания пресета."""

    # Имя владельца параметров для понятных сообщений об ошибках.
    owner: str
    # Сырые значения параметров из TOML-секции или другого config-источника.
    values: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Скопировать входные значения, чтобы не менять внешний словарь."""
        self.values = dict(self.values)

    def pop_int_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать целочисленный параметр, если он явно задан."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            self._raise_type_error(key=key, expected="integer", value=value)
        target[key] = value

    def pop_float_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать числовой параметр, если он явно задан."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._raise_type_error(key=key, expected="number", value=value, )
        target[key] = float(value)

    def pop_bool_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать булевый параметр, если он явно задан."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        if not isinstance(value, bool):
            self._raise_type_error(key=key, expected="boolean", value=value)
        target[key] = value

    def pop_str_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать строковый параметр, если он явно задан."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        if not isinstance(value, str):
            self._raise_type_error(key=key, expected="string", value=value)
        target[key] = value

    def pop_int_tuple_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать последовательность целых чисел, если она явно задана."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        values = self._ensure_sequence(key=key, value=value)
        result: list[int] = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, int):
                self._raise_sequence_type_error(key=key, expected="integer", value=item)
            result.append(item)
        target[key] = tuple(result)

    def pop_float_tuple_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать последовательность чисел, если она явно задана."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        values = self._ensure_sequence(key=key, value=value)
        result: list[float] = []
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                self._raise_sequence_type_error(key=key, expected="number", value=item)
            result.append(float(item))
        target[key] = tuple(result)

    def pop_str_tuple_to(self, target: dict[str, object], key: str) -> None:
        """Прочитать последовательность строк, если она явно задана."""
        value = self._pop_optional(key)
        if value is _MISSING:
            return
        values = self._ensure_sequence(key=key, value=value)
        result: list[str] = []
        for item in values:
            if not isinstance(item, str):
                self._raise_sequence_type_error(key=key, expected="string", value=item)
            result.append(item)
        target[key] = tuple(result)

    def ensure_empty(self) -> None:
        """Проверить, что не осталось неизвестных параметров."""
        if not self.values:
            return
        raise RuntimeError( f"Unsupported params for {self.owner}: {tuple(sorted(self.values))}.")

    def _pop_optional(self, key: str) -> object:
        """Извлечь параметр, если он есть во входных значениях."""
        return self.values.pop(key, _MISSING)

    def _ensure_sequence(self, key: str, value: object) -> list[object] | tuple[object, ...]:
        """Проверить, что значение является списком или кортежем."""
        if not isinstance(value, (list, tuple)):
            self._raise_type_error(key=key, expected="array", value=value)
        return value

    def _raise_type_error(self, key: str, expected: str, value: object) -> None:
        """Выбросить ошибку неверного типа параметра."""
        raise RuntimeError(f"{self.owner} param {key!r} must be {expected}, got {type(value).__name__}.")

    def _raise_sequence_type_error(self, key: str, expected: str, value: object) -> None:
        """Выбросить ошибку неверного типа элемента последовательности."""
        raise RuntimeError(f"{self.owner} param {key!r} items must be {expected}, got {type(value).__name__}.")
