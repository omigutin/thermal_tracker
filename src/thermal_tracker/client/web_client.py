"""Запуск браузерного клиента Thermal Tracker."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import tomllib
import webbrowser

from thermal_tracker.core.config import PROJECT_ROOT

DEFAULT_WEB_CLIENT_CONFIG = "configs/web_client.toml"


@dataclass(frozen=True)
class WebClientConfig:
    """Настройки запуска web-клиента."""

    server_url: str = "http://127.0.0.1:8080"
    open_browser: bool = True


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер web-клиента."""

    parser = argparse.ArgumentParser(description="Открывает Web UI серверного gateway.")
    parser.add_argument("--config", default=DEFAULT_WEB_CLIENT_CONFIG, help="Путь к TOML-конфигу web-клиента.")
    parser.add_argument("--server-url", default="", help="URL запущенного сервера, если нужно переопределить конфиг.")
    parser.add_argument("--no-open", action="store_true", help="Только напечатать URL, не открывать браузер.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Открывает web-клиент в браузере или печатает адрес."""

    args = build_argument_parser().parse_args(argv)
    config = load_web_client_config(args.config)
    url = (args.server_url or config.server_url).rstrip("/")
    open_browser = config.open_browser and not args.no_open

    print(f"Web UI: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)


def load_web_client_config(config_path: str | Path = DEFAULT_WEB_CLIENT_CONFIG) -> WebClientConfig:
    """Загружает настройки web-клиента из рабочего конфига или шаблона."""

    path = _resolve_web_client_config_path(config_path)
    with path.open("rb") as file:
        data = tomllib.load(file)
    client = data.get("client")
    if not isinstance(client, dict):
        client = {}
    return WebClientConfig(
        server_url=str(client.get("server_url") or WebClientConfig.server_url),
        open_browser=bool(client.get("open_browser", WebClientConfig.open_browser)),
    )


def _resolve_web_client_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(__file__).parent / "web" / "web_client.toml"


if __name__ == "__main__":
    main()
