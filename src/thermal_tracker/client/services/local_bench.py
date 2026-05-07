"""Локальный запуск полного стенда gateway + runtime + sender."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from thermal_tracker.core.connections.shared_memory import DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, DEFAULT_SHARED_MEMORY_PREFIX
from thermal_tracker.server.services.shared_memory_cleanup import cleanup_shared_memory


@dataclass(frozen=True)
class _ProcessSpec:
    """Команда одного процесса стенда."""

    name: str
    command: list[str]


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер локального стенда."""

    parser = argparse.ArgumentParser(description="Запускает полный локальный стенд thermal_tracker.")
    parser.add_argument("--prefix", default=DEFAULT_SHARED_MEMORY_PREFIX, help="Префикс Shared Memory.")
    parser.add_argument("--host", default="127.0.0.1", help="Host gateway.")
    parser.add_argument("--port", type=int, default=8080, help="Port gateway.")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Ширина RAW Y8 кадра.")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Высота RAW Y8 кадра.")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS synthetic/video sender.")
    parser.add_argument("--config", default="tests/fixtures/server_smoke.toml", help="Runtime config.")
    parser.add_argument("--video-path", default="", help="Если задано, вместо synthetic sender используется видео.")
    parser.add_argument("--no-cleanup", action="store_true", help="Не удалять старые Shared Memory сегменты перед стартом.")
    parser.add_argument("--metrics-log", default="", help="JSONL лог runtime-результатов.")
    parser.add_argument("--ingress-log", default="", help="JSONL лог входящих gateway-кадров.")
    return parser


def main() -> None:
    """Точка входа `python -m thermal_tracker.client.services.local_bench`."""

    args = build_argument_parser().parse_args()
    if not args.no_cleanup:
        cleanup_shared_memory(prefix=args.prefix, camera_count=4)

    gateway_url = f"http://127.0.0.1:{args.port}"
    process_specs = _build_process_specs(args, gateway_url)
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for spec in process_specs:
            print(f"[bench] start {spec.name}: {' '.join(spec.command)}", flush=True)
            processes.append((spec.name, subprocess.Popen(spec.command)))
            if spec.name == "gateway":
                time.sleep(1.2)
            elif spec.name == "runtime":
                time.sleep(0.4)

        print(f"[bench] Web UI: {gateway_url}", flush=True)
        while all(process.poll() is None for _, process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[bench] stop requested", flush=True)
    finally:
        _stop_processes(processes)


def _build_process_specs(args: argparse.Namespace, gateway_url: str) -> list[_ProcessSpec]:
    """Собирает команды процессов локального стенда."""

    gateway_command = [
        sys.executable,
        "-m",
        "thermal_tracker.server.services.gateway_service",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--prefix",
        args.prefix,
    ]
    if args.ingress_log:
        gateway_command.extend(["--ingress-log", args.ingress_log])

    runtime_command = [
        sys.executable,
        "-m",
        "thermal_tracker.server.services.shared_memory_runtime_worker",
        "--config",
        args.config,
        "--report-every",
        "50",
        "--prefix",
        args.prefix,
    ]
    if args.metrics_log:
        runtime_command.extend(["--metrics-log", args.metrics_log])

    if args.video_path:
        sender_command = [
            sys.executable,
            "-m",
            "thermal_tracker.client.services.network_video_sender",
            args.video_path,
            "--gateway-url",
            gateway_url,
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--loop",
        ]
    else:
        sender_command = [
            sys.executable,
            "-m",
            "thermal_tracker.client.services.synthetic_network_sender",
            "--gateway-url",
            gateway_url,
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]

    return [
        _ProcessSpec("gateway", gateway_command),
        _ProcessSpec("runtime", runtime_command),
        _ProcessSpec("sender", sender_command),
    ]


def _stop_processes(processes: list[tuple[str, subprocess.Popen]]) -> None:
    """Останавливает процессы стенда без жёсткого убийства на первом шаге."""

    for _, process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5.0
    for name, process in reversed(processes):
        if process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[bench] kill {name}", flush=True)
            process.kill()


if __name__ == "__main__":
    main()
