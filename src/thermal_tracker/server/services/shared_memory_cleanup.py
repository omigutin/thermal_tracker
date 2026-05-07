"""Утилита очистки сегментов Shared Memory стенда."""

from __future__ import annotations

import argparse
from multiprocessing import shared_memory

from thermal_tracker.core.connections.shared_memory import DEFAULT_SHARED_MEMORY_PREFIX
from thermal_tracker.core.connections.shared_memory.protocol import frame_buffer_name, json_buffer_name


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер очистки Shared Memory."""

    parser = argparse.ArgumentParser(description="Удаляет Shared Memory сегменты thermal_tracker.")
    parser.add_argument("--prefix", default=DEFAULT_SHARED_MEMORY_PREFIX, help="Префикс сегментов Shared Memory.")
    parser.add_argument("--camera-count", type=int, default=4, help="Сколько camera_id проверить.")
    return parser


def cleanup_shared_memory(*, prefix: str = DEFAULT_SHARED_MEMORY_PREFIX, camera_count: int = 4) -> list[str]:
    """Удаляет известные сегменты Shared Memory и возвращает список удалённых имён."""

    removed: list[str] = []
    names = []
    for camera_id in range(max(1, int(camera_count))):
        names.append(frame_buffer_name(prefix, camera_id, "frame_meta"))
        names.append(frame_buffer_name(prefix, camera_id, "frame_data"))
    names.append(json_buffer_name(prefix, "commands"))
    names.append(json_buffer_name(prefix, "results"))

    for name in names:
        try:
            segment = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            continue
        try:
            segment.close()
            segment.unlink()
            removed.append(name)
        except FileNotFoundError:
            pass
    return removed


def main() -> None:
    """Точка входа `python -m thermal_tracker.server.services.shared_memory_cleanup`."""

    args = build_argument_parser().parse_args()
    removed = cleanup_shared_memory(prefix=args.prefix, camera_count=args.camera_count)
    if removed:
        print("Удалены Shared Memory сегменты:")
        for name in removed:
            print(f"- {name}")
    else:
        print("Shared Memory сегменты для удаления не найдены.")


if __name__ == "__main__":
    main()
