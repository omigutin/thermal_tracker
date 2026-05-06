"""Локальный симулятор источника RAW Y8 кадров для Shared Memory.

Он читает обычный видеофайл, переводит кадры в grayscale uint8 и пишет
последний кадр в Shared Memory так, будто это внешний источник камеры.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ..connections.shared_memory import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
    SharedMemoryFrameBuffer,
)
from ..connections.shared_memory.protocol import now_ns


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер симулятора кадров."""

    parser = argparse.ArgumentParser(description="Пишет кадры видео в Shared Memory как RAW Y8.")
    parser.add_argument("video_path", help="Путь к видеофайлу.")
    parser.add_argument("--prefix", default=DEFAULT_SHARED_MEMORY_PREFIX, help="Префикс сегментов Shared Memory.")
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID, help="ID камеры.")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Ширина RAW Y8 кадра.")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Высота RAW Y8 кадра.")
    parser.add_argument("--fps", type=float, default=25.0, help="Скорость публикации кадров.")
    parser.add_argument("--loop", action="store_true", help="Зациклить видео.")
    parser.add_argument("--max-frames", type=int, default=0, help="Остановиться после N кадров, 0 значит без лимита.")
    parser.add_argument("--report-every", type=int, default=100, help="Печатать статистику каждые N кадров.")
    return parser


def run_simulator(
    *,
    video_path: str | Path,
    prefix: str = DEFAULT_SHARED_MEMORY_PREFIX,
    camera_id: int = DEFAULT_CAMERA_ID,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    fps: float = 25.0,
    loop: bool = False,
    max_frames: int = 0,
    report_every: int = 100,
) -> None:
    """Публикует кадры видео в однослотовый Shared Memory буфер."""

    source_path = Path(video_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source_path}")

    buffer = SharedMemoryFrameBuffer(
        prefix=prefix,
        camera_id=camera_id,
        width=width,
        height=height,
        channels=1,
        frame_format="raw_y8",
        create=True,
    )
    frame_interval = 1.0 / max(1e-6, float(fps))
    published = 0
    frame_id = 0
    report_started = time.perf_counter()
    try:
        while max_frames <= 0 or published < max_frames:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                if not loop:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            if gray.shape[1] != width or gray.shape[0] != height:
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

            frame_id += 1
            buffer.write_frame(gray, frame_id=frame_id, timestamp_ns=now_ns())
            published += 1
            if report_every > 0 and published % report_every == 0:
                elapsed_report = max(time.perf_counter() - report_started, 1e-6)
                print(f"published={published} fps={report_every / elapsed_report:.2f}", flush=True)
                report_started = time.perf_counter()

            elapsed = time.perf_counter() - started
            sleep_seconds = frame_interval - elapsed
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        capture.release()
        buffer.close()


def main() -> None:
    """Точка входа `python -m thermal_tracker.services.shared_memory_video_simulator`."""

    args = build_argument_parser().parse_args()
    run_simulator(
        video_path=args.video_path,
        prefix=args.prefix,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        loop=args.loop,
        max_frames=args.max_frames,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
