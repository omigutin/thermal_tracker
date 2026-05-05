"""Конфиги запуска dev/runtime, загружаемые из TOML-файлов."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
T = TypeVar("T")


@dataclass(frozen=True)
class AppConfig:
    mode: str = "dev"
    scenario: str = "nn_manual"
    preset: str = ""


@dataclass(frozen=True)
class FrameConnectionConfig:
    reader: str = "opencv_video"
    source_path: str = "video"
    camera_count: int = 1


@dataclass(frozen=True)
class CommandConnectionConfig:
    reader: str = "gui"


@dataclass(frozen=True)
class ResultConnectionConfig:
    writer: str = "gui"


@dataclass(frozen=True)
class ConnectionsConfig:
    frames: FrameConnectionConfig
    commands: CommandConnectionConfig
    results: ResultConnectionConfig


@dataclass(frozen=True)
class GuiConfig:
    enabled: bool = True
    show_technical_info: bool = True
    record_output: bool = False
    output_dir: str = "out"


@dataclass(frozen=True)
class ModelConfig:
    type: str = "yolo"
    model_path: str = "models/model.pt"


@dataclass(frozen=True)
class TrackerRunConfig:
    config_path: str = "trackers/botsort.yaml"


@dataclass(frozen=True)
class HistoryConfig:
    enabled: bool = False
    store: str = "null"


@dataclass(frozen=True)
class RuntimeConfig:
    app: AppConfig
    connections: ConnectionsConfig
    gui: GuiConfig
    model: ModelConfig
    tracker: TrackerRunConfig
    history: HistoryConfig
    source_path: Path


def _resolve_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _as_table(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _build_dataclass(cls: type[T], values: dict[str, object]) -> T:
    allowed = {field.name for field in fields(cls)}
    clean = {key: value for key, value in values.items() if key in allowed}
    return cls(**clean)


def load_app_config(config_path: str | Path) -> RuntimeConfig:
    path = _resolve_path(config_path)
    with path.open("rb") as file:
        data = tomllib.load(file)

    connections = _as_table(data.get("connections"))
    return RuntimeConfig(
        app=_build_dataclass(AppConfig, _as_table(data.get("app"))),
        connections=ConnectionsConfig(
            frames=_build_dataclass(FrameConnectionConfig, _as_table(connections.get("frames"))),
            commands=_build_dataclass(CommandConnectionConfig, _as_table(connections.get("commands"))),
            results=_build_dataclass(ResultConnectionConfig, _as_table(connections.get("results"))),
        ),
        gui=_build_dataclass(GuiConfig, _as_table(data.get("gui"))),
        model=_build_dataclass(ModelConfig, _as_table(data.get("model"))),
        tracker=_build_dataclass(TrackerRunConfig, _as_table(data.get("tracker"))),
        history=_build_dataclass(HistoryConfig, _as_table(data.get("history"))),
        source_path=path,
    )
