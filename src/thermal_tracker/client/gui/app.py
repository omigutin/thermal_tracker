"""Точка входа GUI для dev-режима."""

from __future__ import annotations

from pathlib import Path

from thermal_tracker.core.config import AVAILABLE_PRESETS, DEFAULT_PRESET_NAME, PROJECT_ROOT, load_app_config
from thermal_tracker.core.scenarios.scenario_factory import default_preset_for_scenario
from .video_workspace_window import TrackingPlayerWindow


def _default_video_from_config(source_path: str) -> str:
    path = Path(source_path)
    if path.is_file():
        return str(path)
    return ""


def _resolve_gui_config_path(config_path: str) -> str:
    path = Path(config_path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path(__file__).with_name("desktop_client.toml"))


def run_gui(
    *,
    config_path: str = "configs/desktop_client.toml",
    default_video: str = "",
    default_preset: str = "",
    default_delay_ms: int = 30,
    auto_start: bool = False,
) -> None:
    """Открывает интерактивный GUI с настройками из run-конфига."""

    app_config = load_app_config(_resolve_gui_config_path(config_path))
    preset_name = default_preset or app_config.app.preset or default_preset_for_scenario(app_config.app.scenario)
    if preset_name not in AVAILABLE_PRESETS:
        preset_name = DEFAULT_PRESET_NAME

    video_path = default_video or _default_video_from_config(app_config.connections.frames.source_path)
    player = TrackingPlayerWindow(
        default_video=video_path,
        default_preset=preset_name,
        default_delay_ms=default_delay_ms,
        auto_start=auto_start and bool(video_path),
    )
    player.run()


main = run_gui


def build_argument_parser():
    """Возвращает CLI-парсер из `cli.py` для совместимости."""

    from ..cli import build_argument_parser as _build_argument_parser

    return _build_argument_parser()
