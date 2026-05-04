"""Headless runtime application assembly."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig, load_app_config
from .connections.commands import create_command_reader
from .connections.frames import create_frame_reader
from .connections.results import create_result_writer
from .storage import create_history_store


@dataclass
class RuntimeApp:
    config: RuntimeConfig
    scenario_name: str
    scenario: object | None
    frame_reader: object
    command_reader: object
    result_writer: object
    history_store: object

    def initialize_scenario(self) -> object:
        if self.scenario is None:
            from .scenarios import ScenarioFactory

            self.scenario = ScenarioFactory.create(self.scenario_name)
        return self.scenario

    def close(self) -> None:
        for component in (self.frame_reader, self.command_reader, self.result_writer, self.history_store):
            close = getattr(component, "close", None)
            if callable(close):
                close()


def build_runtime_app(config_path: str = "presets/runtime.toml", *, initialize_scenario: bool = False) -> RuntimeApp:
    config = load_app_config(config_path)
    app = RuntimeApp(
        config=config,
        scenario_name=config.app.scenario,
        scenario=None,
        frame_reader=create_frame_reader(config.connections.frames),
        command_reader=create_command_reader(config.connections.commands),
        result_writer=create_result_writer(config.connections.results),
        history_store=create_history_store(config.history),
    )
    if initialize_scenario:
        app.initialize_scenario()
    return app


def run_runtime(config_path: str = "presets/runtime.toml", *, initialize_scenario: bool = False) -> RuntimeApp:
    """Build the headless runtime app from config.

    The shared-memory readers/writers are placeholders for now, so this function
    intentionally assembles the runtime without starting a blind processing loop.
    """

    return build_runtime_app(config_path, initialize_scenario=initialize_scenario)
