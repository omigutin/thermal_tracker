"""Сборка runtime-приложения без GUI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

from thermal_tracker.core.config import RuntimeConfig, TrackerPreset, build_preset, load_app_config
from thermal_tracker.core.connections.commands import create_command_reader
from thermal_tracker.core.connections.frames import create_frame_reader
from thermal_tracker.core.connections.results import create_result_writer
from thermal_tracker.core.connections.shared_memory import SharedMemoryFrame
from thermal_tracker.core.connections.shared_memory.protocol import now_ns
from thermal_tracker.core.domain.models import BoundingBox, TrackSnapshot
from thermal_tracker.core.stages.frame_stabilization.result import FrameStabilizerResult
from thermal_tracker.core.domain.runtime import ScenarioStepResult, SessionRuntimeState
from thermal_tracker.core.diagnostic_writer import create_diagnostic_builder
from thermal_tracker.core.scenarios import default_preset_for_scenario
from thermal_tracker.core.storage import create_history_store


@dataclass
class RuntimeApp:
    config: RuntimeConfig
    scenario_name: str
    scenario: object | None
    frame_reader: object
    command_reader: object
    result_writer: object
    history_store: object
    runtime_state: SessionRuntimeState = field(default_factory=SessionRuntimeState)
    preset_name_override: str | None = None
    neural_model_path_override: str | None = None
    queued_runtime_command: object | None = None

    def __post_init__(self) -> None:
        """Создает билдер диагностики по флагу из runtime-конфига."""

        self._diagnostic_builder = create_diagnostic_builder(self.config.diagnostics.enabled)

    def initialize_scenario(self) -> object:
        if self.scenario is None:
            from thermal_tracker.core.scenarios import ScenarioFactory

            preset_name = self.preset_name_override or self.config.app.preset or None
            preset_override = self._build_preset_override(preset_name) if self.neural_model_path_override else None
            self.scenario = ScenarioFactory.create(
                self.scenario_name,
                preset_name=preset_name,
                preset_override=preset_override,
            )
        return self.scenario

    def close(self) -> None:
        for component in (self.frame_reader, self.command_reader, self.result_writer, self.history_store):
            close = getattr(component, "close", None)
            if callable(close):
                close()

    def process_once(self) -> dict[str, object] | None:
        """Обрабатывает один новый кадр из подключенного источника."""

        command = self._read_command()
        if command is not None:
            self.queued_runtime_command = command

        ok, raw_frame = self.frame_reader.read()
        if not ok or raw_frame is None:
            return None

        self._apply_runtime_command(self.queued_runtime_command)
        self.queued_runtime_command = None

        scenario = self.initialize_scenario()
        frame_metadata = getattr(self.frame_reader, "last_frame", None)
        started_ns = now_ns()
        frame_id = int(getattr(frame_metadata, "frame_id", self.runtime_state.frame_index + 1))
        self._diagnostic_builder.start_frame(frame_id)
        step_result = scenario.process_next_raw_frame(raw_frame, self.runtime_state)
        diagnostics = self._diagnostic_builder.finalize_frame()
        finished_ns = now_ns()
        payload = self._build_result_payload(step_result, frame_metadata, started_ns, finished_ns, diagnostics)
        self.result_writer.write(payload)
        return payload

    def run_loop(self, *, idle_sleep_seconds: float = 0.001, stop_after_frames: int | None = None) -> None:
        """Запускает простой runtime-цикл до остановки процесса."""

        processed_frames = 0
        try:
            while stop_after_frames is None or processed_frames < stop_after_frames:
                result = self.process_once()
                if result is None:
                    time.sleep(max(0.0, idle_sleep_seconds))
                    continue
                processed_frames += 1
        finally:
            self.close()

    def _read_command(self) -> object | None:
        """Читает команду, если command-reader её предоставил."""

        read = getattr(self.command_reader, "read", None)
        if not callable(read):
            return None
        return read()

    def _apply_runtime_command(self, command: object | None) -> None:
        """Перекладывает сетевую/Shared Memory команду в состояние сценария."""

        if not isinstance(command, dict):
            return

        command_type = str(command.get("type") or command.get("command") or "").strip().lower()
        if command_type == "contrast_component":
            try:
                x = int(command["x"])
                y = int(command["y"])
            except (KeyError, TypeError, ValueError):
                return
            self.runtime_state.pending_click = (x, y)
            return

        if command_type in {"reset", "clear"}:
            self.runtime_state.reset_requested = True
            return

        if command_type in {"configure", "set_preset"}:
            scenario = str(command.get("scenario") or "").strip()
            preset_name = str(command.get("preset_name") or command.get("preset") or "").strip()
            model_path = str(command.get("model_path") or "").strip()
            if scenario:
                self.scenario_name = scenario
            if preset_name:
                self.preset_name_override = preset_name
            elif scenario:
                self.preset_name_override = default_preset_for_scenario(scenario)
            self.neural_model_path_override = model_path or None
            self.scenario = None
            self.runtime_state = SessionRuntimeState()

    def _build_preset_override(self, preset_name: str | None) -> TrackerPreset | None:
        """Собирает пресет с переопределённым путём модели для NN-сценария."""

        if not preset_name or not self.neural_model_path_override:
            return None

        preset = build_preset(preset_name)
        if preset.neural is None:
            return None

        return replace(
            preset,
            neural=replace(preset.neural, model_path=self.neural_model_path_override),
        )

    def _build_result_payload(
        self,
        step_result: ScenarioStepResult,
        frame_metadata: SharedMemoryFrame | None,
        started_ns: int,
        finished_ns: int,
        diagnostics: object | None,
    ) -> dict[str, object]:
        """Собирает JSON-результат runtime-шагa для Shared Memory/Web."""

        snapshot = step_result.snapshot
        payload: dict[str, object] = {
            "scenario": self.scenario_name,
            "processing_ms": (finished_ns - started_ns) / 1_000_000.0,
            "runtime_started_ns": started_ns,
            "runtime_finished_ns": finished_ns,
            "snapshot": _snapshot_to_dict(snapshot),
            "frame_shape": list(step_result.frame.bgr.shape),
        }
        if diagnostics is not None:
            payload["diagnostics"] = diagnostics.to_dict()

        if frame_metadata is not None:
            payload.update(
                {
                    "frame_id": frame_metadata.frame_id,
                    "camera_id": frame_metadata.camera_id,
                    "frame_timestamp_ns": frame_metadata.timestamp_ns,
                    "frame_written_ns": frame_metadata.written_ns,
                    "frame_sequence": frame_metadata.sequence,
                    "ingress_to_runtime_ms": (started_ns - frame_metadata.written_ns) / 1_000_000.0,
                    "source_to_runtime_ms": (started_ns - frame_metadata.timestamp_ns) / 1_000_000.0,
                    "source_to_result_ms": (finished_ns - frame_metadata.timestamp_ns) / 1_000_000.0,
                }
            )
        return payload


def build_runtime_app(config_path: str = "configs/server.toml", *, initialize_scenario: bool = False) -> RuntimeApp:
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


def run_runtime(config_path: str = "configs/server.toml", *, initialize_scenario: bool = False) -> RuntimeApp:
    """Собирает runtime-приложение по конфигу.

    Функция намеренно не запускает бесконечный цикл сама: его включает внешний
    сервис или диагностический скрипт, когда источники кадров уже подняты.
    """

    return build_runtime_app(config_path, initialize_scenario=initialize_scenario)


def _bbox_to_dict(bbox: BoundingBox | None) -> dict[str, int] | None:
    """Преобразует bbox в JSON-совместимый словарь."""

    if bbox is None:
        return None
    return {
        "x": int(bbox.x),
        "y": int(bbox.y),
        "width": int(bbox.width),
        "height": int(bbox.height),
    }


def _motion_to_dict(motion: FrameStabilizerResult) -> dict[str, object]:
    """Преобразует оценку движения камеры в JSON-совместимый словарь."""

    return {
        "dx": float(motion.dx),
        "dy": float(motion.dy),
        "response": float(motion.response),
        "valid": bool(motion.valid),
    }


def _snapshot_to_dict(snapshot: TrackSnapshot) -> dict[str, object]:
    """Преобразует снимок трекера в компактный JSON."""

    return {
        "state": snapshot.state.value,
        "track_id": snapshot.track_id,
        "bbox": _bbox_to_dict(snapshot.bbox),
        "predicted_bbox": _bbox_to_dict(snapshot.predicted_bbox),
        "search_region": _bbox_to_dict(snapshot.search_region),
        "score": float(snapshot.score),
        "lost_frames": int(snapshot.lost_frames),
        "global_motion": _motion_to_dict(snapshot.global_motion),
        "message": snapshot.message,
    }
