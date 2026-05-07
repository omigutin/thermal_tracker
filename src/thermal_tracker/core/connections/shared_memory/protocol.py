"""Константы и маленькие утилиты протокола Shared Memory."""

from __future__ import annotations

import time

DEFAULT_SHARED_MEMORY_PREFIX = "thermal_tracker"
DEFAULT_CAMERA_ID = 0
DEFAULT_FRAME_WIDTH = 512
DEFAULT_FRAME_HEIGHT = 640
DEFAULT_FRAME_CHANNELS = 1
FRAME_FORMAT_RAW_Y8 = "raw_y8"
DEFAULT_FRAME_FORMAT = FRAME_FORMAT_RAW_Y8

DTYPE_UINT8 = "uint8"
FRAME_DTYPE_CODE_UINT8 = 1
FRAME_FORMAT_CODE_RAW_Y8 = 1
READY_FLAG = 1

MAX_JSON_PAYLOAD_BYTES = 64 * 1024


def now_ns() -> int:
    """Возвращает монотонное время для замеров задержек."""

    return time.monotonic_ns()


def frame_dtype_to_code(dtype: str) -> int:
    """Преобразует строковый dtype в компактный код протокола."""

    normalized = dtype.strip().lower()
    if normalized != DTYPE_UINT8:
        raise ValueError(f"Shared Memory сейчас поддерживает только uint8, получено: {dtype!r}")
    return FRAME_DTYPE_CODE_UINT8


def frame_dtype_from_code(code: int) -> str:
    """Преобразует код dtype обратно в строку."""

    if int(code) == FRAME_DTYPE_CODE_UINT8:
        return DTYPE_UINT8
    raise ValueError(f"Неизвестный dtype-код кадра: {code!r}")


def frame_format_to_code(frame_format: str) -> int:
    """Преобразует формат кадра в компактный код протокола."""

    normalized = frame_format.strip().lower()
    if normalized != FRAME_FORMAT_RAW_Y8:
        raise ValueError(f"Shared Memory сейчас поддерживает только raw_y8, получено: {frame_format!r}")
    return FRAME_FORMAT_CODE_RAW_Y8


def frame_format_from_code(code: int) -> str:
    """Преобразует код формата кадра обратно в строку."""

    if int(code) == FRAME_FORMAT_CODE_RAW_Y8:
        return FRAME_FORMAT_RAW_Y8
    raise ValueError(f"Неизвестный код формата кадра: {code!r}")


def frame_buffer_name(prefix: str, camera_id: int, kind: str) -> str:
    """Строит имя сегмента Shared Memory для кадров."""

    return f"{prefix}_cam{int(camera_id)}_{kind}"


def json_buffer_name(prefix: str, kind: str) -> str:
    """Строит имя сегмента Shared Memory для JSON-канала."""

    return f"{prefix}_{kind}"
