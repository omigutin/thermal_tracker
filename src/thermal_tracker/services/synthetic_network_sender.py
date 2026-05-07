"""Синтетический сетевой источник RAW Y8 для проверки стенда без видеофайлов."""

from __future__ import annotations

import argparse
import math
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from ..connections.shared_memory import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
from ..connections.shared_memory.protocol import now_ns


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер синтетического отправителя."""

    parser = argparse.ArgumentParser(description="Генерирует RAW Y8 кадры и отправляет их в gateway.")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080", help="URL gateway-сервиса.")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Ширина RAW Y8 кадра.")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Высота RAW Y8 кадра.")
    parser.add_argument("--fps", type=float, default=25.0, help="Скорость отправки кадров.")
    parser.add_argument("--max-frames", type=int, default=0, help="Остановиться после N кадров, 0 значит без лимита.")
    parser.add_argument("--target-size", type=int, default=18, help="Размер яркой цели в пикселях.")
    parser.add_argument("--speed-x", type=float, default=1.8, help="Горизонтальная скорость цели, пикселей/кадр.")
    parser.add_argument("--speed-y", type=float, default=0.35, help="Вертикальная скорость цели, пикселей/кадр.")
    parser.add_argument("--noise", type=float, default=3.0, help="Стандартное отклонение шума фона.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout в секундах.")
    parser.add_argument("--report-every", type=int, default=100, help="Печатать статистику каждые N кадров.")
    return parser


def run_sender(
    *,
    gateway_url: str,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
    fps: float = 25.0,
    max_frames: int = 0,
    target_size: int = 18,
    speed_x: float = 1.8,
    speed_y: float = 0.35,
    noise: float = 3.0,
    timeout: float = 5.0,
    report_every: int = 100,
) -> None:
    """Генерирует движущуюся цель и отправляет кадры в gateway."""

    endpoint = gateway_url.rstrip("/") + "/api/frames/raw-y8"
    frame_interval = 1.0 / max(1e-6, float(fps))
    rng = np.random.default_rng(42)
    sent = 0
    http_latency_sum_ms = 0.0
    report_started = time.perf_counter()

    while max_frames <= 0 or sent < max_frames:
        started = time.perf_counter()
        frame_id = sent + 1
        frame = _build_frame(
            frame_id=frame_id,
            width=width,
            height=height,
            target_size=target_size,
            speed_x=speed_x,
            speed_y=speed_y,
            noise=noise,
            rng=rng,
        )
        http_latency_ms = _post_frame(
            endpoint,
            frame.tobytes(),
            frame_id=frame_id,
            timestamp_ns=now_ns(),
            timeout=timeout,
        )
        http_latency_sum_ms += http_latency_ms
        sent += 1

        if report_every > 0 and sent % report_every == 0:
            elapsed_report = max(time.perf_counter() - report_started, 1e-6)
            print(
                f"synthetic_sent={sent} fps={report_every / elapsed_report:.2f} "
                f"avg_http_post_ms={http_latency_sum_ms / report_every:.2f}",
                flush=True,
            )
            report_started = time.perf_counter()
            http_latency_sum_ms = 0.0

        elapsed = time.perf_counter() - started
        sleep_seconds = frame_interval - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _build_frame(
    *,
    frame_id: int,
    width: int,
    height: int,
    target_size: int,
    speed_x: float,
    speed_y: float,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Строит один синтетический thermal-like кадр."""

    frame = np.full((height, width), 36, dtype=np.float32)
    if noise > 0:
        frame += rng.normal(0.0, noise, size=frame.shape).astype(np.float32)

    wave = math.sin(frame_id * 0.035)
    x = int(round(30 + frame_id * speed_x)) % max(1, width - target_size - 30)
    y_center = height * 0.45 + frame_id * speed_y + wave * 18.0
    y = int(round(y_center)) % max(1, height - target_size - 30)
    y = max(15, min(height - target_size - 15, y))

    cv2.rectangle(frame, (x, y), (x + target_size, y + target_size), 215, -1)
    cv2.circle(frame, (x + target_size // 2, y + target_size // 2), max(2, target_size // 4), 245, -1)
    cv2.putText(frame, str(frame_id), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 90, 1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _post_frame(endpoint: str, payload: bytes, *, frame_id: int, timestamp_ns: int, timeout: float) -> float:
    """Отправляет один RAW Y8 кадр HTTP POST-ом."""

    query = urlencode({"frame_id": frame_id, "timestamp_ns": timestamp_ns})
    request = Request(
        f"{endpoint}?{query}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"Gateway вернул HTTP {response.status}.")
    return (time.perf_counter() - started) * 1000.0


def main() -> None:
    """Точка входа `python -m thermal_tracker.services.synthetic_network_sender`."""

    args = build_argument_parser().parse_args()
    run_sender(
        gateway_url=args.gateway_url,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_frames=args.max_frames,
        target_size=args.target_size,
        speed_x=args.speed_x,
        speed_y=args.speed_y,
        noise=args.noise,
        timeout=args.timeout,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
