"""Буфер последнего RAW Y8 кадра в Shared Memory."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import struct

import numpy as np

from .protocol import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_CHANNELS,
    DEFAULT_FRAME_FORMAT,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
    DTYPE_UINT8,
    READY_FLAG,
    frame_buffer_name,
    frame_dtype_from_code,
    frame_dtype_to_code,
    frame_format_from_code,
    frame_format_to_code,
    now_ns,
)

_FRAME_MAGIC = b"TTFRM001"
_FRAME_VERSION = 1
_FRAME_HEADER = struct.Struct("<8s H H I Q Q I I I H H H H I Q Q Q")


@dataclass(frozen=True)
class SharedMemoryFrame:
    """Кадр и метаданные, прочитанные из Shared Memory."""

    image: np.ndarray
    frame_id: int
    camera_id: int
    timestamp_ns: int
    written_ns: int
    sequence: int
    dropped_frames: int
    frame_format: str
    dtype: str


@dataclass(frozen=True)
class _FrameHeader:
    """Внутреннее представление заголовка кадрового буфера."""

    sequence: int
    frame_id: int
    camera_id: int
    width: int
    height: int
    channels: int
    dtype_code: int
    format_code: int
    flags: int
    payload_size: int
    timestamp_ns: int
    written_ns: int
    dropped_frames: int

    @property
    def ready(self) -> bool:
        """Проверяет, завершена ли запись кадра."""

        return bool(self.flags & READY_FLAG) and self.sequence > 0 and self.sequence % 2 == 0


class SharedMemoryFrameBuffer:
    """Однослотовый буфер кадра: writer перезаписывает, reader берёт последний."""

    def __init__(
        self,
        *,
        prefix: str = DEFAULT_SHARED_MEMORY_PREFIX,
        camera_id: int = DEFAULT_CAMERA_ID,
        width: int = DEFAULT_FRAME_WIDTH,
        height: int = DEFAULT_FRAME_HEIGHT,
        channels: int = DEFAULT_FRAME_CHANNELS,
        frame_format: str = DEFAULT_FRAME_FORMAT,
        create: bool = False,
        create_if_missing: bool = False,
    ) -> None:
        self.prefix = prefix
        self.camera_id = int(camera_id)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.frame_format = frame_format
        self.dtype = DTYPE_UINT8
        self.max_payload_size = self.width * self.height * self.channels
        self.meta_name = frame_buffer_name(prefix, self.camera_id, "frame_meta")
        self.data_name = frame_buffer_name(prefix, self.camera_id, "frame_data")
        self._meta = self._open_segment(self.meta_name, _FRAME_HEADER.size, create, create_if_missing)
        self._data = self._open_segment(self.data_name, self.max_payload_size, create, create_if_missing)
        if create or create_if_missing:
            self._write_empty_header()

    @classmethod
    def try_open(
        cls,
        *,
        prefix: str = DEFAULT_SHARED_MEMORY_PREFIX,
        camera_id: int = DEFAULT_CAMERA_ID,
        width: int = DEFAULT_FRAME_WIDTH,
        height: int = DEFAULT_FRAME_HEIGHT,
        channels: int = DEFAULT_FRAME_CHANNELS,
        frame_format: str = DEFAULT_FRAME_FORMAT,
    ) -> "SharedMemoryFrameBuffer | None":
        """Открывает существующий буфер и возвращает `None`, если его ещё нет."""

        try:
            return cls(
                prefix=prefix,
                camera_id=camera_id,
                width=width,
                height=height,
                channels=channels,
                frame_format=frame_format,
                create=False,
                create_if_missing=False,
            )
        except FileNotFoundError:
            return None

    @staticmethod
    def _open_segment(
        name: str,
        size: int,
        create: bool,
        create_if_missing: bool,
    ) -> shared_memory.SharedMemory:
        """Открывает или создаёт один сегмент Shared Memory."""

        try:
            return shared_memory.SharedMemory(name=name, create=create, size=size)
        except FileExistsError:
            return shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            if not create_if_missing:
                raise
            segment = shared_memory.SharedMemory(name=name, create=True, size=size)
            return segment

    def write_frame(
        self,
        frame: np.ndarray,
        *,
        frame_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> SharedMemoryFrame:
        """Записывает кадр в буфер и возвращает итоговые метаданные."""

        prepared = self._prepare_frame(frame)
        payload_size = int(prepared.nbytes)
        if payload_size > self.max_payload_size:
            raise ValueError(
                f"Кадр занимает {payload_size} байт, буфер рассчитан на {self.max_payload_size} байт."
            )

        previous = self._read_header(allow_empty=True)
        previous_sequence = previous.sequence if previous is not None else 0
        writing_sequence = previous_sequence + 1
        if writing_sequence % 2 == 0:
            writing_sequence += 1
        final_sequence = writing_sequence + 1
        next_frame_id = int(frame_id) if frame_id is not None else (previous.frame_id + 1 if previous else 1)
        source_timestamp_ns = int(timestamp_ns) if timestamp_ns is not None else now_ns()

        self._write_header(
            _FrameHeader(
                sequence=writing_sequence,
                frame_id=next_frame_id,
                camera_id=self.camera_id,
                width=prepared.shape[1],
                height=prepared.shape[0],
                channels=self._channels_for(prepared),
                dtype_code=frame_dtype_to_code(self.dtype),
                format_code=frame_format_to_code(self.frame_format),
                flags=0,
                payload_size=payload_size,
                timestamp_ns=source_timestamp_ns,
                written_ns=now_ns(),
                dropped_frames=0,
            )
        )
        self._data.buf[:payload_size] = prepared.reshape(-1).tobytes()
        header = _FrameHeader(
            sequence=final_sequence,
            frame_id=next_frame_id,
            camera_id=self.camera_id,
            width=prepared.shape[1],
            height=prepared.shape[0],
            channels=self._channels_for(prepared),
            dtype_code=frame_dtype_to_code(self.dtype),
            format_code=frame_format_to_code(self.frame_format),
            flags=READY_FLAG,
            payload_size=payload_size,
            timestamp_ns=source_timestamp_ns,
            written_ns=now_ns(),
            dropped_frames=0,
        )
        self._write_header(header)
        return self._frame_from_header(header)

    def read_latest(self, *, after_sequence: int | None = None) -> SharedMemoryFrame | None:
        """Читает последний завершённый кадр, если он новый."""

        for _ in range(3):
            header = self._read_header(allow_empty=True)
            if header is None or not header.ready:
                return None
            if after_sequence is not None and header.sequence <= after_sequence:
                return None

            frame = self._frame_from_header(header)
            check_header = self._read_header(allow_empty=True)
            if check_header is not None and check_header.sequence == header.sequence and check_header.ready:
                return frame
        return None

    def close(self) -> None:
        """Закрывает локальные handle-ы Shared Memory."""

        self._meta.close()
        self._data.close()

    def unlink(self) -> None:
        """Удаляет сегменты Shared Memory. Вызывать должен владелец буферов."""

        for segment in (self._meta, self._data):
            try:
                segment.unlink()
            except FileNotFoundError:
                pass

    def _write_empty_header(self) -> None:
        """Инициализирует заголовок пустым состоянием."""

        self._write_header(
            _FrameHeader(
                sequence=0,
                frame_id=0,
                camera_id=self.camera_id,
                width=self.width,
                height=self.height,
                channels=self.channels,
                dtype_code=frame_dtype_to_code(self.dtype),
                format_code=frame_format_to_code(self.frame_format),
                flags=0,
                payload_size=0,
                timestamp_ns=0,
                written_ns=0,
                dropped_frames=0,
            )
        )

    def _write_header(self, header: _FrameHeader) -> None:
        """Пишет заголовок в meta-сегмент."""

        _FRAME_HEADER.pack_into(
            self._meta.buf,
            0,
            _FRAME_MAGIC,
            _FRAME_VERSION,
            _FRAME_HEADER.size,
            self.max_payload_size,
            int(header.sequence),
            int(header.frame_id),
            int(header.camera_id),
            int(header.width),
            int(header.height),
            int(header.channels),
            int(header.dtype_code),
            int(header.format_code),
            int(header.flags),
            int(header.payload_size),
            int(header.timestamp_ns),
            int(header.written_ns),
            int(header.dropped_frames),
        )

    def _read_header(self, *, allow_empty: bool = False) -> _FrameHeader | None:
        """Читает заголовок и проверяет базовую совместимость."""

        unpacked = _FRAME_HEADER.unpack_from(self._meta.buf, 0)
        magic = unpacked[0]
        if magic == b"\x00" * len(_FRAME_MAGIC) and allow_empty:
            return None
        if magic != _FRAME_MAGIC:
            raise RuntimeError(f"Сегмент {self.meta_name!r} не похож на кадровый буфер thermal_tracker.")
        version = int(unpacked[1])
        if version != _FRAME_VERSION:
            raise RuntimeError(f"Неподдерживаемая версия кадрового протокола: {version}.")
        return _FrameHeader(
            sequence=int(unpacked[4]),
            frame_id=int(unpacked[5]),
            camera_id=int(unpacked[6]),
            width=int(unpacked[7]),
            height=int(unpacked[8]),
            channels=int(unpacked[9]),
            dtype_code=int(unpacked[10]),
            format_code=int(unpacked[11]),
            flags=int(unpacked[12]),
            payload_size=int(unpacked[13]),
            timestamp_ns=int(unpacked[14]),
            written_ns=int(unpacked[15]),
            dropped_frames=int(unpacked[16]),
        )

    def _frame_from_header(self, header: _FrameHeader) -> SharedMemoryFrame:
        """Копирует кадр из data-сегмента по уже прочитанному заголовку."""

        dtype = frame_dtype_from_code(header.dtype_code)
        frame_format = frame_format_from_code(header.format_code)
        if dtype != DTYPE_UINT8:
            raise ValueError(f"Неподдерживаемый dtype кадра: {dtype!r}")

        flat = np.ndarray((header.payload_size,), dtype=np.uint8, buffer=self._data.buf[:header.payload_size]).copy()
        if header.channels == 1:
            image = flat.reshape((header.height, header.width))
        else:
            image = flat.reshape((header.height, header.width, header.channels))
        return SharedMemoryFrame(
            image=image,
            frame_id=header.frame_id,
            camera_id=header.camera_id,
            timestamp_ns=header.timestamp_ns,
            written_ns=header.written_ns,
            sequence=header.sequence,
            dropped_frames=header.dropped_frames,
            frame_format=frame_format,
            dtype=dtype,
        )

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Проверяет RAW Y8 кадр и приводит его к непрерывному uint8-массиву."""

        if frame.dtype != np.uint8:
            raise ValueError(f"RAW Y8 должен быть uint8, получено: {frame.dtype}.")
        if frame.ndim == 2:
            channels = 1
        elif frame.ndim == 3:
            channels = frame.shape[2]
        else:
            raise ValueError(f"Неподдерживаемая размерность кадра: {frame.shape!r}.")
        if channels != self.channels:
            raise ValueError(f"Ожидалось каналов: {self.channels}, получено: {channels}.")
        if frame.shape[0] > self.height or frame.shape[1] > self.width:
            raise ValueError(
                f"Кадр {frame.shape[1]}x{frame.shape[0]} больше буфера {self.width}x{self.height}."
            )
        return np.ascontiguousarray(frame)

    @staticmethod
    def _channels_for(frame: np.ndarray) -> int:
        """Возвращает число каналов кадра."""

        return 1 if frame.ndim == 2 else int(frame.shape[2])
