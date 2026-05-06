"""Читатель команд из Shared Memory."""

from __future__ import annotations

from ..shared_memory import DEFAULT_SHARED_MEMORY_PREFIX, SharedMemoryJsonBuffer
from .base_command_reader import BaseCommandReader


class SharedMemoryCommandReader(BaseCommandReader):
    implementation_name = "shared_memory"
    is_ready = True

    def __init__(self, *, prefix: str = DEFAULT_SHARED_MEMORY_PREFIX) -> None:
        self.prefix = prefix
        self._buffer: SharedMemoryJsonBuffer | None = None
        self._last_message_id = 0

    def read(self):
        """Читает последнюю новую команду, если она есть."""

        buffer = self._ensure_buffer()
        if buffer is None:
            return None

        message = buffer.read_latest(after_message_id=self._last_message_id)
        if message is None:
            return None

        self._last_message_id = message.message_id
        return message.payload

    def close(self) -> None:
        """Закрывает подключение к командному буферу."""

        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def _ensure_buffer(self) -> SharedMemoryJsonBuffer | None:
        """Лениво подключается к командному буферу, когда он появится."""

        if self._buffer is not None:
            return self._buffer
        self._buffer = SharedMemoryJsonBuffer.try_open(prefix=self.prefix, kind="commands")
        return self._buffer
