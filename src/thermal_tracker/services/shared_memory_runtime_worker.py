"""Runtime worker для обработки кадров из Shared Memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..runtime_app import run_runtime


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер runtime worker-а."""

    parser = argparse.ArgumentParser(description="Читает кадры из Shared Memory и пишет результаты обратно.")
    parser.add_argument("--config", default="configs/runtime.toml", help="Путь к runtime TOML.")
    parser.add_argument("--idle-sleep", type=float, default=0.001, help="Пауза, когда нового кадра нет.")
    parser.add_argument("--max-frames", type=int, default=0, help="Остановиться после N обработанных кадров.")
    parser.add_argument("--init-scenario", action="store_true", help="Инициализировать сценарий сразу при старте.")
    parser.add_argument("--report-every", type=int, default=50, help="Печатать статистику каждые N кадров.")
    parser.add_argument("--prefix", default="", help="Переопределить Shared Memory prefix из конфига.")
    parser.add_argument("--metrics-log", default="", help="JSONL файл с результатами обработки.")
    return parser


def main() -> None:
    """Точка входа `python -m thermal_tracker.services.shared_memory_runtime_worker`."""

    args = build_argument_parser().parse_args()
    app = run_runtime(args.config, initialize_scenario=args.init_scenario)
    if args.prefix:
        _override_shared_memory_prefix(app, args.prefix)
    metrics_log_file = _open_metrics_log(args.metrics_log)
    processed = 0
    report_started = time.perf_counter()
    try:
        while args.max_frames <= 0 or processed < args.max_frames:
            result = app.process_once()
            if result is None:
                time.sleep(max(0.0, args.idle_sleep))
                continue

            processed += 1
            if metrics_log_file is not None:
                metrics_log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                metrics_log_file.flush()
            if args.report_every > 0 and processed % args.report_every == 0:
                elapsed_report = max(time.perf_counter() - report_started, 1e-6)
                processing_ms = float(result.get("processing_ms", 0.0))
                source_to_result_ms = float(result.get("source_to_result_ms", 0.0))
                ingress_to_runtime_ms = float(result.get("ingress_to_runtime_ms", 0.0))
                frame_id = result.get("frame_id", "?")
                print(
                    "processed="
                    f"{processed} frame_id={frame_id} fps={args.report_every / elapsed_report:.2f} "
                    f"processing_ms={processing_ms:.2f} "
                    f"ingress_to_runtime_ms={ingress_to_runtime_ms:.2f} "
                    f"source_to_result_ms={source_to_result_ms:.2f}",
                    flush=True,
                )
                report_started = time.perf_counter()
    finally:
        if metrics_log_file is not None:
            metrics_log_file.close()
        app.close()


def _override_shared_memory_prefix(app, prefix: str) -> None:
    """Переопределяет prefix у lazy Shared Memory адаптеров до первого чтения."""

    for component in (app.frame_reader, app.command_reader, app.result_writer):
        if hasattr(component, "prefix"):
            setattr(component, "prefix", prefix)


def _open_metrics_log(path: str):
    """Открывает JSONL-лог runtime-метрик, если путь задан."""

    clean_path = path.strip()
    if not clean_path:
        return None
    target = Path(clean_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("a", encoding="utf-8")


if __name__ == "__main__":
    main()
