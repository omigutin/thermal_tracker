"""Командная строка серверной стороны трекера."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from thermal_tracker.core.config import PROJECT_ROOT, load_app_config
from thermal_tracker.core.connections.shared_memory import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
)
from thermal_tracker.server.services.gateway_service import GatewayConfig, run_gateway
from thermal_tracker.server.services.shared_memory_cleanup import cleanup_shared_memory
from thermal_tracker.server.services.shared_memory_runtime_worker import run_worker

DEFAULT_SERVER_CONFIG = "configs/server.toml"
SERVER_MODES = ("all", "gateway", "runtime", "cleanup")


@dataclass(frozen=True)
class _ProcessSpec:
    """Описание одного дочернего процесса серверного стенда."""

    name: str
    command: list[str]


@dataclass(frozen=True)
class _ServerSettings:
    """Настройки серверного запуска после чтения TOML."""

    mode: str = "all"
    prefix: str = DEFAULT_SHARED_MEMORY_PREFIX
    camera_id: int = DEFAULT_CAMERA_ID
    camera_count: int = 4
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    host: str = "0.0.0.0"
    port: int = 8080
    config_path: str = DEFAULT_SERVER_CONFIG
    idle_sleep: float = 0.001
    max_frames: int = 0
    init_scenario: bool = False
    report_every: int = 50
    ingress_log: str = ""
    metrics_log: str = ""
    cleanup_before_start: bool = False


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер серверного запуска."""

    parser = argparse.ArgumentParser(description="Запуск серверной части Thermal Tracker.")
    parser.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=SERVER_MODES,
        help="Режим: all запускает gateway и runtime worker.",
    )
    parser.add_argument("--prefix", default=None, help="Префикс сегментов Shared Memory.")
    parser.add_argument("--camera-id", type=int, default=None, help="ID камеры.")
    parser.add_argument("--camera-count", type=int, default=None, help="Количество камер для cleanup.")
    parser.add_argument("--width", type=int, default=None, help="Ширина RAW Y8 кадра.")
    parser.add_argument("--height", type=int, default=None, help="Высота RAW Y8 кадра.")
    parser.add_argument("--host", default=None, help="Адрес HTTP gateway.")
    parser.add_argument("--port", type=int, default=None, help="Порт HTTP gateway.")
    parser.add_argument("--config", default=DEFAULT_SERVER_CONFIG, help="Путь к server TOML-конфигу.")
    parser.add_argument("--idle-sleep", type=float, default=None, help="Пауза worker, если новых кадров нет.")
    parser.add_argument("--max-frames", type=int, default=None, help="Остановить runtime после N кадров, 0 = без лимита.")
    parser.add_argument("--init-scenario", action="store_true", default=None, help="Инициализировать сценарий сразу при старте.")
    parser.add_argument("--report-every", type=int, default=None, help="Печатать runtime-метрики каждые N кадров.")
    parser.add_argument("--ingress-log", default=None, help="JSONL лог входящих кадров gateway.")
    parser.add_argument("--metrics-log", default=None, help="JSONL лог результатов runtime.")
    parser.add_argument(
        "--cleanup-before-start",
        action="store_true",
        default=None,
        help="Удалить старые Shared Memory сегменты перед запуском режима all.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Запускает выбранный серверный режим."""

    parser = build_argument_parser()
    args = _merge_arguments_with_config(parser.parse_args(argv), parser)
    if args.mode == "gateway":
        _run_gateway_mode(args)
        return
    if args.mode == "runtime":
        _run_runtime_mode(args)
        return
    if args.mode == "cleanup":
        _run_cleanup_mode(args)
        return
    _run_all_mode(args)


def _merge_arguments_with_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Дополняет CLI-аргументы значениями из server.toml."""

    settings = _load_server_settings(args.config)
    args.mode = _pick(args.mode, settings.mode)
    args.prefix = _pick(args.prefix, settings.prefix)
    args.camera_id = _pick(args.camera_id, settings.camera_id)
    args.camera_count = _pick(args.camera_count, settings.camera_count)
    args.width = _pick(args.width, settings.width)
    args.height = _pick(args.height, settings.height)
    args.host = _pick(args.host, settings.host)
    args.port = _pick(args.port, settings.port)
    args.config = settings.config_path
    args.idle_sleep = _pick(args.idle_sleep, settings.idle_sleep)
    args.max_frames = _pick(args.max_frames, settings.max_frames)
    args.init_scenario = _pick(args.init_scenario, settings.init_scenario)
    args.report_every = _pick(args.report_every, settings.report_every)
    args.ingress_log = _pick(args.ingress_log, settings.ingress_log)
    args.metrics_log = _pick(args.metrics_log, settings.metrics_log)
    args.cleanup_before_start = _pick(args.cleanup_before_start, settings.cleanup_before_start)

    if args.mode not in SERVER_MODES:
        parser.error(f"Неподдерживаемый режим сервера в конфиге: {args.mode!r}")
    return args


def _load_server_settings(config_path: str | Path) -> _ServerSettings:
    """Читает серверный TOML и вытаскивает настройки процессов."""

    resolved_path = _resolve_server_config_path(config_path)
    with resolved_path.open("rb") as file:
        raw_config = tomllib.load(file)

    app_config = load_app_config(resolved_path)
    frames = app_config.connections.frames
    server = _as_table(raw_config.get("server"))
    return _ServerSettings(
        mode=_as_str(server.get("mode"), "all"),
        prefix=frames.shared_memory_prefix or DEFAULT_SHARED_MEMORY_PREFIX,
        camera_id=frames.camera_id,
        camera_count=frames.camera_count,
        width=frames.frame_width,
        height=frames.frame_height,
        host=_as_str(server.get("host"), "0.0.0.0"),
        port=_as_int(server.get("port"), 8080),
        config_path=str(resolved_path),
        idle_sleep=_as_float(server.get("idle_sleep"), 0.001),
        max_frames=_as_int(server.get("max_frames"), 0),
        init_scenario=_as_bool(server.get("init_scenario"), False),
        report_every=_as_int(server.get("report_every"), 50),
        ingress_log=_as_str(server.get("ingress_log"), ""),
        metrics_log=_as_str(server.get("metrics_log"), ""),
        cleanup_before_start=_as_bool(server.get("cleanup_before_start"), False),
    )


def _resolve_server_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(__file__).with_name("server.toml")


def _run_gateway_mode(args: argparse.Namespace) -> None:
    """Запускает только HTTP gateway."""

    run_gateway(
        GatewayConfig(
            prefix=args.prefix,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            host=args.host,
            port=args.port,
            ingress_log_path=args.ingress_log,
        )
    )


def _run_runtime_mode(args: argparse.Namespace) -> None:
    """Запускает только runtime worker."""

    run_worker(
        config_path=args.config,
        idle_sleep_seconds=args.idle_sleep,
        max_frames=args.max_frames,
        init_scenario=args.init_scenario,
        report_every=args.report_every,
        prefix=args.prefix,
        metrics_log_path=args.metrics_log,
    )


def _run_cleanup_mode(args: argparse.Namespace) -> None:
    """Удаляет известные Shared Memory сегменты."""

    removed = cleanup_shared_memory(prefix=args.prefix, camera_count=args.camera_count)
    if removed:
        print("Удалены Shared Memory сегменты:", flush=True)
        for name in removed:
            print(f"- {name}", flush=True)
        return
    print("Shared Memory сегменты для удаления не найдены.", flush=True)


def _run_all_mode(args: argparse.Namespace) -> None:
    """Запускает gateway и runtime worker как два дочерних процесса."""

    if args.cleanup_before_start:
        cleanup_shared_memory(prefix=args.prefix, camera_count=args.camera_count)

    process_specs = _build_process_specs(args)
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for spec in process_specs:
            print(f"[server] start {spec.name}: {' '.join(spec.command)}", flush=True)
            processes.append((spec.name, subprocess.Popen(spec.command)))
            if spec.name == "gateway":
                time.sleep(1.0)

        web_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"[server] Web UI: http://{web_host}:{args.port}", flush=True)
        while all(process.poll() is None for _, process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[server] stop requested", flush=True)
    finally:
        _stop_processes(processes)


def _build_process_specs(args: argparse.Namespace) -> list[_ProcessSpec]:
    """Собирает команды дочерних процессов режима all."""

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
        "--camera-id",
        str(args.camera_id),
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
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-frames",
        str(args.max_frames),
        "--report-every",
        str(args.report_every),
        "--prefix",
        args.prefix,
    ]
    if args.init_scenario:
        runtime_command.append("--init-scenario")
    if args.metrics_log:
        runtime_command.extend(["--metrics-log", args.metrics_log])

    return [
        _ProcessSpec("gateway", gateway_command),
        _ProcessSpec("runtime", runtime_command),
    ]


def _stop_processes(processes: list[tuple[str, subprocess.Popen]]) -> None:
    """Останавливает дочерние процессы серверного запуска."""

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
            print(f"[server] kill {name}", flush=True)
            process.kill()


def _pick(value, fallback):
    return fallback if value is None else value


def _as_table(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _as_str(value: object, fallback: str) -> str:
    return fallback if value is None else str(value)


def _as_int(value: object, fallback: int) -> int:
    try:
        return fallback if value is None else int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: object, fallback: float) -> float:
    try:
        return fallback if value is None else float(value)
    except (TypeError, ValueError):
        return fallback


def _as_bool(value: object, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


if __name__ == "__main__":
    main()
