"""Писатель результатов в Shared Memory."""

from __future__ import annotations

from dataclasses import is_dataclass
from enum import Enum
from typing import Any

from ..shared_memory import DEFAULT_SHARED_MEMORY_PREFIX, SharedMemoryJsonBuffer
from .base_result_writer import BaseResultWriter


class SharedMemoryResultWriter(BaseResultWriter):
    implementation_name = "shared_memory"
    is_ready = True

    def __init__(self, *, prefix: str = DEFAULT_SHARED_MEMORY_PREFIX, create_if_missing: bool = True) -> None:
        self.prefix = prefix
        self.create_if_missing = create_if_missing
        self._buffer: SharedMemoryJsonBuffer | None = None

    def write(self, result: Any) -> None:
        """Записывает результат обработки в JSON-буфер."""

        buffer = self._ensure_buffer()
        if buffer is None:
            return
        buffer.write_message(_to_jsonable_result(result))

    def close(self) -> None:
        """Закрывает подключение к буферу результатов."""

        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def _ensure_buffer(self) -> SharedMemoryJsonBuffer | None:
        """Лениво открывает или создаёт буфер результатов."""

        if self._buffer is not None:
            return self._buffer
        self._buffer = SharedMemoryJsonBuffer(
            prefix=self.prefix,
            kind="results",
            create=False,
            create_if_missing=self.create_if_missing,
        )
        return self._buffer


def _to_jsonable_result(result: Any) -> dict[str, Any]:
    """Превращает результат runtime в JSON-совместимый словарь."""

    if isinstance(result, dict):
        return _to_jsonable_value(result)

    snapshot = getattr(result, "snapshot", None)
    if snapshot is not None:
        payload: dict[str, Any] = {"snapshot": _to_jsonable_value(snapshot)}
        frame = getattr(result, "frame", None)
        if frame is not None:
            raw_frame = getattr(frame, "bgr", None)
            if raw_frame is not None:
                payload["frame_shape"] = list(raw_frame.shape)
        return payload

    return {"value": _to_jsonable_value(result)}


def _to_jsonable_value(value: Any) -> Any:
    """Рекурсивно приводит объект к JSON-совместимому виду."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _to_jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_value(item) for item in value]
    if is_dataclass(value):
        return {
            field_name: _to_jsonable_value(getattr(value, field_name))
            for field_name in getattr(value, "__dataclass_fields__", {})
        }
    if hasattr(value, "to_xywh"):
        x, y, width, height = value.to_xywh()
        return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}
    return str(value)
