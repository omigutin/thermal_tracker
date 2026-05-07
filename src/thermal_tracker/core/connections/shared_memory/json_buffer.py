"""JSON-канал поверх одного сегмента Shared Memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from multiprocessing import shared_memory
import struct
from typing import Any

from .protocol import DEFAULT_SHARED_MEMORY_PREFIX, MAX_JSON_PAYLOAD_BYTES, READY_FLAG, json_buffer_name, now_ns

_JSON_MAGIC = b"TTJSON01"
_JSON_VERSION = 1
_JSON_HEADER = struct.Struct("<8s H H I Q Q I Q Q H H")


@dataclass(frozen=True)
class SharedMemoryJsonMessage:
    """Сообщение, прочитанное из JSON-канала."""

    payload: dict[str, Any]
    message_id: int
    timestamp_ns: int
    written_ns: int
    sequence: int


class SharedMemoryJsonBuffer:
    """Однослотовый JSON-буфер для команд или результатов."""

    def __init__(
        self,
        *,
        prefix: str = DEFAULT_SHARED_MEMORY_PREFIX,
        kind: str,
        max_payload_bytes: int = MAX_JSON_PAYLOAD_BYTES,
        create: bool = False,
        create_if_missing: bool = False,
    ) -> None:
        self.prefix = prefix
        self.kind = kind
        self.max_payload_bytes = int(max_payload_bytes)
        self.name = json_buffer_name(prefix, kind)
        self._segment = self._open_segment(create=create, create_if_missing=create_if_missing)
        if create:
            self._write_empty_header()

    @classmethod
    def try_open(
        cls,
        *,
        prefix: str = DEFAULT_SHARED_MEMORY_PREFIX,
        kind: str,
        max_payload_bytes: int = MAX_JSON_PAYLOAD_BYTES,
    ) -> "SharedMemoryJsonBuffer | None":
        """Открывает существующий JSON-буфер или возвращает `None`."""

        try:
            return cls(
                prefix=prefix,
                kind=kind,
                max_payload_bytes=max_payload_bytes,
                create=False,
                create_if_missing=False,
            )
        except FileNotFoundError:
            return None

    def write_message(
        self,
        payload: dict[str, Any],
        *,
        message_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> SharedMemoryJsonMessage:
        """Записывает JSON-сообщение в буфер."""

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.max_payload_bytes:
            raise ValueError(
                f"JSON-сообщение занимает {len(encoded)} байт, буфер рассчитан на {self.max_payload_bytes} байт."
            )

        current = self._read_header(allow_empty=True)
        previous_sequence = current[0] if current is not None else 0
        previous_message_id = current[1] if current is not None else 0
        writing_sequence = previous_sequence + 1
        if writing_sequence % 2 == 0:
            writing_sequence += 1
        final_sequence = writing_sequence + 1
        next_message_id = int(message_id) if message_id is not None else previous_message_id + 1
        source_timestamp_ns = int(timestamp_ns) if timestamp_ns is not None else now_ns()

        self._write_header(
            sequence=writing_sequence,
            message_id=next_message_id,
            payload_size=0,
            timestamp_ns=source_timestamp_ns,
            written_ns=now_ns(),
            flags=0,
        )
        payload_offset = _JSON_HEADER.size
        self._segment.buf[payload_offset:payload_offset + len(encoded)] = encoded
        self._write_header(
            sequence=final_sequence,
            message_id=next_message_id,
            payload_size=len(encoded),
            timestamp_ns=source_timestamp_ns,
            written_ns=now_ns(),
            flags=READY_FLAG,
        )
        return SharedMemoryJsonMessage(
            payload=payload,
            message_id=next_message_id,
            timestamp_ns=source_timestamp_ns,
            written_ns=now_ns(),
            sequence=final_sequence,
        )

    def read_latest(self, *, after_message_id: int | None = None) -> SharedMemoryJsonMessage | None:
        """Читает последнее завершённое сообщение, если оно новое."""

        for _ in range(3):
            header = self._read_header(allow_empty=True)
            if header is None:
                return None
            sequence, message_id, payload_size, timestamp_ns, written_ns, flags = header
            if not flags & READY_FLAG or sequence <= 0 or sequence % 2 != 0:
                return None
            if after_message_id is not None and message_id <= after_message_id:
                return None
            payload_offset = _JSON_HEADER.size
            raw_payload = bytes(self._segment.buf[payload_offset:payload_offset + payload_size])
            check_header = self._read_header(allow_empty=True)
            if check_header is None or check_header[0] != sequence:
                continue
            decoded = json.loads(raw_payload.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("JSON-канал ожидает объект верхнего уровня.")
            return SharedMemoryJsonMessage(
                payload=decoded,
                message_id=message_id,
                timestamp_ns=timestamp_ns,
                written_ns=written_ns,
                sequence=sequence,
            )
        return None

    def close(self) -> None:
        """Закрывает локальный handle Shared Memory."""

        self._segment.close()

    def unlink(self) -> None:
        """Удаляет сегмент Shared Memory. Вызывать должен владелец буфера."""

        try:
            self._segment.unlink()
        except FileNotFoundError:
            pass

    def _open_segment(self, *, create: bool, create_if_missing: bool) -> shared_memory.SharedMemory:
        """Открывает или создаёт сегмент JSON-канала."""

        size = _JSON_HEADER.size + self.max_payload_bytes
        try:
            return shared_memory.SharedMemory(name=self.name, create=create, size=size)
        except FileExistsError:
            return shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError:
            if not create_if_missing:
                raise
            segment = shared_memory.SharedMemory(name=self.name, create=True, size=size)
            self._segment = segment
            self._write_empty_header()
            return segment

    def _write_empty_header(self) -> None:
        """Инициализирует пустой JSON-канал."""

        self._write_header(
            sequence=0,
            message_id=0,
            payload_size=0,
            timestamp_ns=0,
            written_ns=0,
            flags=0,
        )

    def _write_header(
        self,
        *,
        sequence: int,
        message_id: int,
        payload_size: int,
        timestamp_ns: int,
        written_ns: int,
        flags: int,
    ) -> None:
        """Пишет заголовок JSON-канала."""

        _JSON_HEADER.pack_into(
            self._segment.buf,
            0,
            _JSON_MAGIC,
            _JSON_VERSION,
            _JSON_HEADER.size,
            self.max_payload_bytes,
            int(sequence),
            int(message_id),
            int(payload_size),
            int(timestamp_ns),
            int(written_ns),
            int(flags),
            0,
        )

    def _read_header(self, *, allow_empty: bool) -> tuple[int, int, int, int, int, int] | None:
        """Читает заголовок JSON-канала."""

        unpacked = _JSON_HEADER.unpack_from(self._segment.buf, 0)
        magic = unpacked[0]
        if magic == b"\x00" * len(_JSON_MAGIC) and allow_empty:
            return None
        if magic != _JSON_MAGIC:
            raise RuntimeError(f"Сегмент {self.name!r} не похож на JSON-буфер thermal_tracker.")
        version = int(unpacked[1])
        if version != _JSON_VERSION:
            raise RuntimeError(f"Неподдерживаемая версия JSON-протокола: {version}.")
        return (
            int(unpacked[4]),
            int(unpacked[5]),
            int(unpacked[6]),
            int(unpacked[7]),
            int(unpacked[8]),
            int(unpacked[9]),
        )
