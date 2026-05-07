"""Командная строка для запуска нового тепловизионного трекера."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .gui.app import run_gui as launch_gui
from thermal_tracker.core.config import AVAILABLE_PRESETS


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(
        description="Запуск GUI тепловизионного трекера с параметрами из командной строки."
    )
    parser.add_argument("--video", help="Путь к видеофайлу, который нужно открыть сразу")
    parser.add_argument(
        "--preset",
        default="",
        choices=AVAILABLE_PRESETS,
        help="Имя пресета трекера",
    )
    parser.add_argument("--delay-ms", type=int, default=30, help="Задержка воспроизведения в миллисекундах")
    parser.add_argument("--config", default="configs/desktop_client.toml", help="Путь к TOML-конфигу desktop-клиента.")
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    return build_argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Запускает GUI с параметрами, пришедшими из командной строки."""

    args = parse_arguments(argv)
    launch_gui(
        config_path=args.config,
        default_video=args.video or "",
        default_preset=args.preset,
        default_delay_ms=args.delay_ms,
        auto_start=bool(args.video),
    )
