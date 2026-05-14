"""HTTP-gateway между сетью, браузером и локальной Shared Memory.

Gateway живёт на той же машине, что и Shared Memory. Внешние устройства
говорят с ним по HTTP, а runtime продолжает читать и писать локальные буферы.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
except ImportError:
    FastAPI = None
    HTTPException = None
    Query = None
    Request = None
    HTMLResponse = None
    JSONResponse = None
    Response = None

from thermal_tracker.core.connections.shared_memory import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
    SharedMemoryFrame,
    SharedMemoryFrameBuffer,
    SharedMemoryJsonBuffer,
)
from thermal_tracker.core.connections.shared_memory.protocol import now_ns
from thermal_tracker.core.config import AVAILABLE_PRESETS, build_preset, get_preset_presentation
from thermal_tracker.core.scenarios.scenario_factory import PIPELINE_KIND_TO_SCENARIO
from thermal_tracker.server.services.web_recording import (
    ContentRect,
    RecordingFrameMetadata,
    WebRecordingManager,
    parse_content_rect,
)


@dataclass
class GatewayConfig:
    """Настройки gateway-сервиса."""

    prefix: str = DEFAULT_SHARED_MEMORY_PREFIX
    camera_id: int = DEFAULT_CAMERA_ID
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    host: str = "0.0.0.0"
    port: int = 8080
    ingress_log_path: str = ""


@dataclass
class GatewayStats:
    """Накопленные диагностические метрики gateway."""

    received_frames: int = 0
    first_received_ns: int = 0
    last_received_ns: int = 0
    last_frame_id: int = 0
    last_payload_bytes: int = 0
    last_http_write_ms: float = 0.0
    last_remote_timestamp_ns: int | None = None


class SharedMemoryGateway:
    """Сервисный объект, который держит Shared Memory буферы и HTTP-логику."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.frame_buffer = SharedMemoryFrameBuffer(
            prefix=config.prefix,
            camera_id=config.camera_id,
            width=config.width,
            height=config.height,
            channels=1,
            frame_format="raw_y8",
            create=False,
            create_if_missing=True,
        )
        self.command_buffer = SharedMemoryJsonBuffer(
            prefix=config.prefix,
            kind="commands",
            create=False,
            create_if_missing=True,
        )
        self._result_buffer: SharedMemoryJsonBuffer | None = None
        self._last_result_message_id = 0
        self._last_result_payload: dict[str, Any] | None = None
        self._stats = GatewayStats()
        self._recent_frame_times_ns: deque[int] = deque(maxlen=120)
        self._ingress_log_file = _open_jsonl_log(config.ingress_log_path)
        self.recording_manager = WebRecordingManager()

    def publish_raw_y8(
        self,
        payload: bytes,
        *,
        frame_id: int | None = None,
        remote_timestamp_ns: int | None = None,
        content_rect: ContentRect | None = None,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Публикует сетевой RAW Y8 кадр в Shared Memory."""

        received_ns = now_ns()
        expected_size = self.config.width * self.config.height
        if len(payload) != expected_size:
            raise ValueError(f"RAW Y8 кадр должен занимать {expected_size} байт, получено {len(payload)}.")

        image = np.frombuffer(payload, dtype=np.uint8).reshape((self.config.height, self.config.width))
        written = self.frame_buffer.write_frame(
            image,
            frame_id=frame_id,
            timestamp_ns=received_ns,
        )
        finished_ns = now_ns()
        self._record_ingress_metrics(
            frame=written,
            received_ns=received_ns,
            finished_ns=finished_ns,
            payload_bytes=len(payload),
            remote_timestamp_ns=remote_timestamp_ns,
        )
        metadata = _frame_metadata_to_dict(written)
        metadata["remote_timestamp_ns"] = remote_timestamp_ns
        metadata["http_write_ms"] = (finished_ns - received_ns) / 1_000_000.0
        metadata["content_rect"] = content_rect.as_dict() if content_rect is not None else None
        metadata["source"] = source or {}
        self._write_ingress_log(metadata)
        result = self.latest_result()
        self.recording_manager.write_frame(
            _draw_result_overlay(written.image, result),
            RecordingFrameMetadata(
                frame=metadata,
                result=result,
                content_rect=content_rect,
                source=source or {},
            ),
        )
        return metadata

    def latest_frame(self) -> SharedMemoryFrame | None:
        """Возвращает последний кадр, даже если он уже был прочитан раньше."""

        return self.frame_buffer.read_latest()

    def latest_result(self) -> dict[str, Any] | None:
        """Возвращает последний результат runtime из Shared Memory."""

        result_buffer = self._ensure_result_buffer()
        if result_buffer is None:
            return self._last_result_payload

        message = result_buffer.read_latest(after_message_id=self._last_result_message_id)
        if message is None:
            return self._last_result_payload

        self._last_result_message_id = message.message_id
        self._last_result_payload = message.payload
        return message.payload

    def write_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Записывает команду оператора в Shared Memory."""

        message = self.command_buffer.write_message(command)
        return {
            "ok": True,
            "message_id": message.message_id,
            "timestamp_ns": message.timestamp_ns,
            "command": command,
        }

    def metrics(self) -> dict[str, Any]:
        """Возвращает диагностическое состояние gateway."""

        frame = self.latest_frame()
        result = self.latest_result()
        current_ns = now_ns()
        frame_payload = _frame_metadata_to_dict(frame) if frame is not None else None
        result_frame_id = _extract_result_frame_id(result)
        latest_frame_id = frame.frame_id if frame is not None else 0
        return {
            "prefix": self.config.prefix,
            "camera_id": self.config.camera_id,
            "width": self.config.width,
            "height": self.config.height,
            "frame": frame_payload,
            "result": result,
            "ingress": self._ingress_metrics(current_ns),
            "lag": {
                "latest_frame_id": latest_frame_id,
                "latest_result_frame_id": result_frame_id,
                "frame_id_lag": max(0, latest_frame_id - result_frame_id) if result_frame_id is not None else None,
                "latest_frame_age_ms": (
                    (current_ns - frame.written_ns) / 1_000_000.0 if frame is not None and frame.written_ns > 0 else None
                ),
                "latest_result_age_ms": _result_age_ms(result, current_ns),
            },
        }

    def close(self) -> None:
        """Закрывает handle-ы Shared Memory."""

        self.frame_buffer.close()
        self.command_buffer.close()
        if self._result_buffer is not None:
            self._result_buffer.close()
            self._result_buffer = None
        if self._ingress_log_file is not None:
            self._ingress_log_file.close()
            self._ingress_log_file = None
        self.recording_manager.stop()

    def _ensure_result_buffer(self) -> SharedMemoryJsonBuffer | None:
        """Лениво подключается к результатам, которые создаёт runtime."""

        if self._result_buffer is not None:
            return self._result_buffer
        self._result_buffer = SharedMemoryJsonBuffer.try_open(prefix=self.config.prefix, kind="results")
        return self._result_buffer

    def _record_ingress_metrics(
        self,
        *,
        frame: SharedMemoryFrame,
        received_ns: int,
        finished_ns: int,
        payload_bytes: int,
        remote_timestamp_ns: int | None,
    ) -> None:
        """Обновляет статистику приёма сетевых кадров."""

        if self._stats.received_frames == 0:
            self._stats.first_received_ns = received_ns
        self._stats.received_frames += 1
        self._stats.last_received_ns = finished_ns
        self._stats.last_frame_id = frame.frame_id
        self._stats.last_payload_bytes = payload_bytes
        self._stats.last_http_write_ms = (finished_ns - received_ns) / 1_000_000.0
        self._stats.last_remote_timestamp_ns = remote_timestamp_ns
        self._recent_frame_times_ns.append(finished_ns)

    def _ingress_metrics(self, current_ns: int) -> dict[str, Any]:
        """Возвращает статистику входного HTTP-потока."""

        uptime_ms = (
            (current_ns - self._stats.first_received_ns) / 1_000_000.0
            if self._stats.first_received_ns > 0
            else 0.0
        )
        if len(self._recent_frame_times_ns) >= 2:
            recent_span_seconds = max(
                (self._recent_frame_times_ns[-1] - self._recent_frame_times_ns[0]) / 1_000_000_000.0,
                1e-6,
            )
            recent_fps = (len(self._recent_frame_times_ns) - 1) / recent_span_seconds
        else:
            recent_fps = 0.0
        return {
            "received_frames": self._stats.received_frames,
            "recent_fps": recent_fps,
            "last_frame_id": self._stats.last_frame_id,
            "last_payload_bytes": self._stats.last_payload_bytes,
            "last_http_write_ms": self._stats.last_http_write_ms,
            "last_remote_timestamp_ns": self._stats.last_remote_timestamp_ns,
            "uptime_ms": uptime_ms,
        }

    def _write_ingress_log(self, metadata: dict[str, Any]) -> None:
        """Пишет событие входного кадра в JSONL, если лог включён."""

        if self._ingress_log_file is None:
            return
        self._ingress_log_file.write(json.dumps({"event": "frame_received", **metadata}, ensure_ascii=False) + "\n")
        self._ingress_log_file.flush()


def create_app(config: GatewayConfig | None = None):
    """Создаёт FastAPI-приложение gateway-сервиса."""

    if FastAPI is None:
        raise RuntimeError("Для gateway нужен FastAPI: установите зависимости через Poetry.")

    gateway = SharedMemoryGateway(config or GatewayConfig())

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            gateway.close()

    app = FastAPI(title="Thermal Tracker Gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _read_web_asset("index.html")

    @app.get("/web/styles.css")
    def web_styles() -> Response:
        return Response(content=_read_web_asset("styles.css"), media_type="text/css; charset=utf-8")

    @app.get("/web/app.js")
    def web_app() -> Response:
        return Response(content=_read_web_asset("app.js"), media_type="text/javascript; charset=utf-8")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "thermal_tracker_gateway", "config": gateway.metrics()}

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.post("/api/frames/raw-y8")
    async def post_raw_y8(
        request: Request,
        frame_id: int | None = Query(default=None),
        timestamp_ns: int | None = Query(default=None),
        content_x: int | None = Query(default=None),
        content_y: int | None = Query(default=None),
        content_width: int | None = Query(default=None),
        content_height: int | None = Query(default=None),
        source_width: int | None = Query(default=None),
        source_height: int | None = Query(default=None),
        source_name: str = Query(default=""),
        preset_name: str = Query(default=""),
    ) -> JSONResponse:
        payload = await request.body()
        content_rect = _content_rect_from_query(content_x, content_y, content_width, content_height)
        source = {
            "name": source_name,
            "width": source_width,
            "height": source_height,
            "preset": preset_name,
        }
        try:
            metadata = gateway.publish_raw_y8(
                payload,
                frame_id=frame_id,
                remote_timestamp_ns=timestamp_ns,
                content_rect=content_rect,
                source=source,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "frame": metadata})

    @app.get("/api/frame/latest.jpg")
    def latest_frame_jpeg(overlay: bool = Query(default=True)) -> Response:
        frame = gateway.latest_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="Кадров в Shared Memory пока нет.")

        image = frame.image
        if overlay:
            image = _draw_result_overlay(image, gateway.latest_result())
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(status_code=500, detail="Не удалось закодировать JPEG.")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    @app.get("/api/frame/latest.raw")
    def latest_frame_raw() -> Response:
        frame = gateway.latest_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="Кадров в Shared Memory пока нет.")
        headers = {
            "X-Frame-Id": str(frame.frame_id),
            "X-Camera-Id": str(frame.camera_id),
            "X-Frame-Width": str(frame.image.shape[1]),
            "X-Frame-Height": str(frame.image.shape[0]),
            "X-Timestamp-Ns": str(frame.timestamp_ns),
        }
        return Response(content=frame.image.tobytes(), media_type="application/octet-stream", headers=headers)

    @app.get("/api/result/latest")
    def latest_result() -> dict[str, Any]:
        return {"result": gateway.latest_result()}

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        return gateway.metrics()

    @app.get("/api/presets")
    def presets() -> dict[str, Any]:
        """Возвращает список доступных пресетов для web-клиента."""

        preset_items = []
        for name in AVAILABLE_PRESETS:
            preset = build_preset(name)
            presentation = get_preset_presentation(name)
            preset_items.append(
                {
                    "name": name,
                    "title": presentation.title,
                    "tooltip": presentation.tooltip,
                    "description": presentation.description,
                    "pipeline_kind": preset.pipeline_kind,
                    "scenario": PIPELINE_KIND_TO_SCENARIO.get(preset.pipeline_kind, "opencv_manual"),
                    "has_neural": preset.neural is not None,
                    "model_path": preset.neural.model_path if preset.neural is not None else "",
                }
            )
        return {
            "presets": preset_items,
        }

    @app.post("/api/commands/configure")
    async def configure(request: Request) -> dict[str, Any]:
        """Передаёт runtime команду сменить сценарий или пресет."""

        data = await request.json()
        preset_name = str(data.get("preset_name") or data.get("preset") or "").strip()
        scenario = str(data.get("scenario") or "").strip()
        command: dict[str, Any] = {"type": "configure", "timestamp_ns": now_ns()}
        if preset_name:
            command["preset_name"] = preset_name
        if scenario:
            command["scenario"] = scenario
        model_path = str(data.get("model_path") or "").strip()
        if model_path:
            command["model_path"] = model_path
        return gateway.write_command(command)

    @app.post("/api/commands/contrast_component")
    async def click(request: Request) -> dict[str, Any]:
        data = await request.json()
        try:
            x = int(data["x"])
            y = int(data["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Команда contrast_component требует x и y.") from exc
        return gateway.write_command({"type": "contrast_component", "x": x, "y": y, "timestamp_ns": now_ns()})

    @app.post("/api/commands/reset")
    def reset() -> dict[str, Any]:
        return gateway.write_command({"type": "reset", "timestamp_ns": now_ns()})

    @app.post("/api/recording/start")
    async def start_recording(request: Request) -> dict[str, Any]:
        """Включает серверную запись видео и JSONL."""

        data = await request.json()
        content_rect = parse_content_rect(data.get("content_rect"))
        frame_size = (
            (content_rect.width, content_rect.height)
            if content_rect is not None
            else (gateway.config.width, gateway.config.height)
        )
        return gateway.recording_manager.start(
            base_name=str(data.get("base_name") or "thermal_tracker_recording"),
            fps=float(data.get("fps") or 25.0),
            frame_size=frame_size,
        )

    @app.post("/api/recording/stop")
    def stop_recording() -> dict[str, Any]:
        """Останавливает серверную запись."""

        return gateway.recording_manager.stop()

    @app.get("/api/recording/status")
    def recording_status() -> dict[str, Any]:
        """Возвращает состояние серверной записи."""

        return gateway.recording_manager.status()

    return app


def build_argument_parser() -> argparse.ArgumentParser:
    """Создаёт CLI-парсер gateway-сервиса."""

    parser = argparse.ArgumentParser(description="HTTP gateway для Shared Memory thermal_tracker.")
    parser.add_argument("--prefix", default=DEFAULT_SHARED_MEMORY_PREFIX, help="Префикс сегментов Shared Memory.")
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID, help="ID камеры.")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Ширина RAW Y8 кадра.")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Высота RAW Y8 кадра.")
    parser.add_argument("--host", default="0.0.0.0", help="Адрес HTTP-сервера.")
    parser.add_argument("--port", type=int, default=8080, help="Порт HTTP-сервера.")
    parser.add_argument("--ingress-log", default="", help="JSONL лог входящих кадров.")
    return parser


def run_gateway(config: GatewayConfig) -> None:
    """Запускает HTTP gateway с переданными настройками."""

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Для запуска gateway нужен uvicorn: установите зависимости через Poetry.") from exc

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )


def main(argv: list[str] | None = None) -> None:
    """Точка входа `python -m thermal_tracker.server.services.gateway_service`."""

    args = build_argument_parser().parse_args(argv)
    config = GatewayConfig(
        prefix=args.prefix,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        host=args.host,
        port=args.port,
        ingress_log_path=args.ingress_log,
    )
    run_gateway(config)


def _frame_metadata_to_dict(frame: SharedMemoryFrame | None) -> dict[str, Any]:
    """Преобразует метаданные кадра в JSON."""

    if frame is None:
        return {}
    return {
        "frame_id": frame.frame_id,
        "camera_id": frame.camera_id,
        "timestamp_ns": frame.timestamp_ns,
        "written_ns": frame.written_ns,
        "sequence": frame.sequence,
        "width": int(frame.image.shape[1]),
        "height": int(frame.image.shape[0]),
        "format": frame.frame_format,
        "dtype": frame.dtype,
    }


def _open_jsonl_log(path: str):
    """Открывает JSONL-лог, если путь задан."""

    clean_path = path.strip()
    if not clean_path:
        return None
    target = Path(clean_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("a", encoding="utf-8")


def _extract_result_frame_id(result: dict[str, Any] | None) -> int | None:
    """Достаёт frame_id последнего результата, если runtime его уже записал."""

    if not isinstance(result, dict):
        return None
    frame_id = result.get("frame_id")
    if isinstance(frame_id, int):
        return frame_id
    try:
        return int(frame_id)
    except (TypeError, ValueError):
        return None


def _result_age_ms(result: dict[str, Any] | None, current_ns: int) -> float | None:
    """Считает возраст результата по локальному времени runtime/gateway."""

    if not isinstance(result, dict):
        return None
    finished_ns = result.get("runtime_finished_ns")
    try:
        finished_ns_int = int(finished_ns)
    except (TypeError, ValueError):
        return None
    if finished_ns_int <= 0:
        return None
    return (current_ns - finished_ns_int) / 1_000_000.0


def _read_web_asset(name: str) -> str:
    """Читает статический файл web-интерфейса из клиентского пакета."""

    return resources.files("thermal_tracker.client.web").joinpath(name).read_text(encoding="utf-8")


def _content_rect_from_query(
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
) -> ContentRect | None:
    """Собирает область реального кадра из query-параметров."""

    if x is None or y is None or width is None or height is None:
        return None
    return ContentRect(x=int(x), y=int(y), width=int(width), height=int(height))


def _draw_result_overlay(image: np.ndarray, result: dict[str, Any] | None) -> np.ndarray:
    """Рисует результат трекера поверх grayscale-кадра для браузера."""

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if not isinstance(result, dict):
        return canvas

    snapshot = result.get("snapshot")
    if not isinstance(snapshot, dict):
        return canvas

    if snapshot.get("state") == "SEARCHING":
        _draw_bbox(canvas, snapshot.get("search_region"), (0, 190, 255))
        _draw_bbox(canvas, snapshot.get("predicted_bbox"), (255, 220, 0))
    target_bbox = _draw_bbox(canvas, snapshot.get("bbox"), (0, 255, 0))
    track_id = snapshot.get("track_id") if isinstance(snapshot, dict) else None
    if track_id is not None and target_bbox is not None:
        x, y, _, _ = target_bbox
        cv2.putText(canvas, f"#{track_id}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return canvas


def _draw_bbox(canvas: np.ndarray, bbox: Any, color: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """Рисует прямоугольник из словаря bbox и возвращает его координаты."""

    if not isinstance(bbox, dict):
        return None
    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 1)
    return x, y, width, height



if __name__ == "__main__":
    main()
