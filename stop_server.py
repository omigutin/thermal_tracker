"""Останавливает runtime-сервер по порту и имени процесса."""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

CONFIG_PATH = Path("configs/server.toml")


@dataclass
class StopReport:
    """Отчет по найденным и остановленным процессам."""

    by_port: set[int] = field(default_factory=set)
    orphaned: set[int] = field(default_factory=set)
    terminated: set[int] = field(default_factory=set)
    killed: set[int] = field(default_factory=set)
    failed: dict[int, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""

    parser = argparse.ArgumentParser(description="Stop thermal_tracker server processes")
    parser.add_argument("--port", type=int, default=None, help="Port to stop (overrides configs/server.toml)")
    return parser.parse_args()


def read_port_from_config() -> int:
    """Читает порт из server.toml."""

    with CONFIG_PATH.open("rb") as file:
        data = tomllib.load(file)
    server_table = data.get("server")
    if not isinstance(server_table, dict):
        raise ValueError("Missing [server] section in configs/server.toml")
    port = server_table.get("port")
    if not isinstance(port, int):
        raise ValueError("server.port must be integer")
    return port


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Запускает системную команду и возвращает результат."""

    return subprocess.run(command, capture_output=True, text=True, check=False)


def find_pids_windows(port: int) -> tuple[set[int], set[int], dict[int, str]]:
    """Ищет PID на Windows по порту и имени процесса."""

    by_port: set[int] = set()
    orphaned: set[int] = set()
    failures: dict[int, str] = {}

    netstat = run_command(["netstat", "-ano"])
    if netstat.returncode == 0:
        pattern = re.compile(rf"^\s*TCP\s+\S+:{port}\s+\S+\s+\S+\s+(\d+)\s*$", re.IGNORECASE)
        for line in netstat.stdout.splitlines():
            match = pattern.match(line)
            if match:
                by_port.add(int(match.group(1)))
    else:
        failures[-1] = f"netstat failed: {netstat.stderr.strip() or netstat.stdout.strip()}"

    tasklist = run_command(["tasklist", "/v", "/fo", "csv"])
    if tasklist.returncode == 0:
        reader = csv.DictReader(tasklist.stdout.splitlines())
        for row in reader:
            fields_text = " ".join(str(row.get(key, "")) for key in ("Image Name", "Window Title"))
            if "thermal_tracker.server" not in fields_text.lower():
                continue
            pid_text = str(row.get("PID", "")).strip()
            if pid_text.isdigit():
                orphaned.add(int(pid_text))
    else:
        failures[-2] = f"tasklist failed: {tasklist.stderr.strip() or tasklist.stdout.strip()}"

    return by_port, orphaned, failures


def find_pids_unix(port: int) -> tuple[set[int], set[int], dict[int, str]]:
    """Ищет PID на Linux/macOS по порту и имени процесса."""

    by_port: set[int] = set()
    orphaned: set[int] = set()
    failures: dict[int, str] = {}

    lsof = run_command(["lsof", "-ti", f":{port}"])
    if lsof.returncode == 0:
        by_port = {int(line.strip()) for line in lsof.stdout.splitlines() if line.strip().isdigit()}
    elif lsof.returncode != 1:
        failures[-1] = f"lsof failed: {lsof.stderr.strip() or lsof.stdout.strip()}"

    pgrep = run_command(["pgrep", "-f", "thermal_tracker.server"])
    if pgrep.returncode == 0:
        orphaned = {int(line.strip()) for line in pgrep.stdout.splitlines() if line.strip().isdigit()}
    elif pgrep.returncode != 1:
        failures[-2] = f"pgrep failed: {pgrep.stderr.strip() or pgrep.stdout.strip()}"

    return by_port, orphaned, failures


def stop_windows(pids: set[int], report: StopReport) -> None:
    """Останавливает процессы на Windows через taskkill."""

    for pid in sorted(pids):
        result = run_command(["taskkill", "/PID", str(pid), "/F"])
        if result.returncode == 0:
            report.killed.add(pid)
        else:
            report.failed[pid] = result.stderr.strip() or result.stdout.strip() or "taskkill failed"


def stop_unix(pids: set[int], report: StopReport) -> None:
    """Останавливает процессы на Unix сначала TERM, потом KILL."""

    alive: set[int] = set()
    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGTERM)
            report.terminated.add(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            report.failed[pid] = f"Permission denied for SIGTERM: {exc}"
        except OSError as exc:
            report.failed[pid] = f"Failed to send SIGTERM: {exc}"

    time.sleep(2.0)
    for pid in sorted(pids):
        if pid in report.failed:
            continue
        try:
            os.kill(pid, 0)
            alive.add(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            report.failed[pid] = f"Permission denied during probe: {exc}"

    for pid in sorted(alive):
        try:
            os.kill(pid, signal.SIGKILL)
            report.killed.add(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            report.failed[pid] = f"Permission denied for SIGKILL: {exc}"
        except OSError as exc:
            report.failed[pid] = f"Failed to send SIGKILL: {exc}"


def print_report(report: StopReport) -> None:
    """Печатает итоговый отчет."""

    print(f"PID by port: {sorted(report.by_port)}")
    print(f"Orphan PID: {sorted(report.orphaned)}")
    print(f"Sent SIGTERM: {sorted(report.terminated)}")
    print(f"Force killed: {sorted(report.killed)}")
    if report.failed:
        print("Failed to stop:")
        for pid, reason in sorted(report.failed.items()):
            print(f"  PID {pid}: {reason}")


def main() -> int:
    """Точка входа скрипта остановки сервера."""

    args = parse_args()
    port = args.port if args.port is not None else read_port_from_config()
    report = StopReport()

    if sys.platform.startswith("win"):
        by_port, orphaned, failures = find_pids_windows(port)
        report.by_port = by_port
        report.orphaned = orphaned
        report.failed.update(failures)
        all_pids = by_port | orphaned
        if not all_pids:
            print("сервер не запущен")
            return 0
        stop_windows(all_pids, report)
    else:
        by_port, orphaned, failures = find_pids_unix(port)
        report.by_port = by_port
        report.orphaned = orphaned
        report.failed.update(failures)
        all_pids = by_port | orphaned
        if not all_pids:
            print("сервер не запущен")
            return 0
        stop_unix(all_pids, report)

    print_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
