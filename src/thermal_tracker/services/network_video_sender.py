"""Сетевой симулятор источника кадров для gateway-сервиса.

Скрипт читает видео, переводит кадры в RAW Y8 нужного размера и отправляет
их HTTP POST-ом в `/api/frames/raw-y8`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2

from ..connections.shared_memory import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
from ..connections.shared_memory.protocol import now_ns


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер сетевого отправителя кадров."""

    parser = argparse.ArgumentParser(description="Отправляет видео в thermal_tracker gateway как RAW Y8.")
    parser.add_argument("video_path", help="Путь к видеофайлу.")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080", help="URL gateway-сервиса.")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Ширина RAW Y8 кадра.")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Высота RAW Y8 кадра.")
    parser.add_argument("--fps", type=float, default=25.0, help="Скорость отправки кадров.")
    parser.add_argument("--loop", action="store_true", help="Зациклить видео.")
    parser.add_argument("--max-frames", type=int, default=0, help="Остановиться после N кадров, 0 значит без лимита.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout в секундах.")
    parser.add_argument("--report-every", type=int, default=100, help="Печатать статистику каждые N кадров.")
    return parser


def run_sender(
    *,
    video_path: str | Path,
    gateway_url: str,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    fps: float = 25.0,
    loop: bool = False,
    max_frames: int = 0,
    timeout: float = 5.0,
    report_every: int = 100,
) -> None:
    """Читает видео и отправляет RAW Y8 кадры в gateway."""

    source_path = Path(video_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source_path}")

    endpoint = gateway_url.rstrip("/") + "/api/frames/raw-y8"
    frame_interval = 1.0 / max(1e-6, float(fps))
    frame_id = 0
    sent = 0
    report_started = time.perf_counter()
    try:
        while max_frames <= 0 or sent < max_frames:
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
            _post_frame(
                endpoint,
                gray.tobytes(),
                frame_id=frame_id,
                timestamp_ns=now_ns(),
                timeout=timeout,
            )
            sent += 1

            if report_every > 0 and sent % report_every == 0:
                elapsed_report = max(time.perf_counter() - report_started, 1e-6)
                print(f"sent={sent} fps={report_every / elapsed_report:.2f}", flush=True)
                report_started = time.perf_counter()

            elapsed = time.perf_counter() - started
            sleep_seconds = frame_interval - elapsed
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        capture.release()


def _post_frame(endpoint: str, payload: bytes, *, frame_id: int, timestamp_ns: int, timeout: float) -> None:
    """Отправляет один RAW Y8 кадр HTTP POST-ом."""

    query = urlencode({"frame_id": frame_id, "timestamp_ns": timestamp_ns})
    request = Request(
        f"{endpoint}?{query}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"Gateway вернул HTTP {response.status}.")


def main() -> None:
    """Точка входа `python -m thermal_tracker.services.network_video_sender`."""

    args = build_argument_parser().parse_args()
    run_sender(
        video_path=args.video_path,
        gateway_url=args.gateway_url,
        width=args.width,
        height=args.height,
        fps=args.fps,
        loop=args.loop,
        max_frames=args.max_frames,
        timeout=args.timeout,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
