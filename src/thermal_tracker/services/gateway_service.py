"""HTTP-gateway между сетью, браузером и локальной Shared Memory.

Gateway живёт на той же машине, что и Shared Memory. Внешние устройства
говорят с ним по HTTP, а runtime продолжает читать и писать локальные буферы.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
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

from ..connections.shared_memory import (
    DEFAULT_CAMERA_ID,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_SHARED_MEMORY_PREFIX,
    SharedMemoryFrame,
    SharedMemoryFrameBuffer,
    SharedMemoryJsonBuffer,
)
from ..connections.shared_memory.protocol import now_ns


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

    def publish_raw_y8(
        self,
        payload: bytes,
        *,
        frame_id: int | None = None,
        remote_timestamp_ns: int | None = None,
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
        self._write_ingress_log(metadata)
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
    app = FastAPI(title="Thermal Tracker Gateway", version="0.1.0")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        gateway.close()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _build_dashboard_html()

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
    ) -> JSONResponse:
        payload = await request.body()
        try:
            metadata = gateway.publish_raw_y8(payload, frame_id=frame_id, remote_timestamp_ns=timestamp_ns)
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

    @app.post("/api/commands/click")
    async def click(request: Request) -> dict[str, Any]:
        data = await request.json()
        try:
            x = int(data["x"])
            y = int(data["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Команда click требует x и y.") from exc
        return gateway.write_command({"type": "click", "x": x, "y": y, "timestamp_ns": now_ns()})

    @app.post("/api/commands/reset")
    def reset() -> dict[str, Any]:
        return gateway.write_command({"type": "reset", "timestamp_ns": now_ns()})

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


def main() -> None:
    """Точка входа `python -m thermal_tracker.services.gateway_service`."""

    args = build_argument_parser().parse_args()
    config = GatewayConfig(
        prefix=args.prefix,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        host=args.host,
        port=args.port,
        ingress_log_path=args.ingress_log,
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Для запуска gateway нужен uvicorn: установите зависимости через Poetry.") from exc

    uvicorn.run(create_app(config), host=config.host, port=config.port)


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


def _draw_result_overlay(image: np.ndarray, result: dict[str, Any] | None) -> np.ndarray:
    """Рисует bbox результата поверх grayscale-кадра для браузера."""

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if not isinstance(result, dict):
        return canvas

    snapshot = result.get("snapshot")
    bbox = snapshot.get("bbox") if isinstance(snapshot, dict) else None
    if not isinstance(bbox, dict):
        return canvas

    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return canvas

    cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 255, 0), 1)
    track_id = snapshot.get("track_id") if isinstance(snapshot, dict) else None
    if track_id is not None:
        cv2.putText(canvas, f"#{track_id}", (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return canvas


def _build_dashboard_html() -> str:
    """Возвращает минимальный диагностический Web UI."""

    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Thermal Tracker Gateway</title>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #111; color: #eee; display: grid; grid-template-columns: 1fr 360px; height: 100vh; }
    main { display: grid; place-items: center; overflow: hidden; position: relative; }
    img { max-width: 100%; max-height: 100vh; image-rendering: pixelated; cursor: crosshair; }
    #empty { color: #aaa; font-size: 18px; text-align: center; padding: 24px; }
    aside { border-left: 1px solid #333; padding: 14px; background: #181818; overflow: auto; }
    button { width: 100%; padding: 9px; margin: 0 0 10px; }
    .metric { display: grid; grid-template-columns: 1fr auto; gap: 8px; border-bottom: 1px solid #2a2a2a; padding: 7px 0; }
    .metric span:first-child { color: #aaa; }
    .metric span:last-child { font-weight: 600; text-align: right; }
    pre { white-space: pre-wrap; font-size: 12px; line-height: 1.35; }
  </style>
</head>
<body>
  <main>
    <div id="empty">Кадров пока нет. Запустите отправитель видео в gateway.</div>
    <img id="frame" alt="latest frame" style="display:none">
  </main>
  <aside>
    <button id="reset">Сбросить трек</button>
    <section id="summary"></section>
    <pre id="metrics">Ожидание кадров...</pre>
  </aside>
  <script>
    const frame = document.getElementById("frame");
    const empty = document.getElementById("empty");
    const metrics = document.getElementById("metrics");
    const summary = document.getElementById("summary");
    let hasFrame = false;
    function valueOrDash(value) {
      return value === undefined || value === null ? "—" : value;
    }
    function numberOrDash(value, digits = 1) {
      return typeof value === "number" ? value.toFixed(digits) : "—";
    }
    function renderSummary(data) {
      const result = data.result || {};
      const snapshot = result.snapshot || {};
      const frameData = data.frame || {};
      const ingress = data.ingress || {};
      const lag = data.lag || {};
      const items = [
        ["frame_id", valueOrDash(frameData.frame_id)],
        ["ingress_fps", numberOrDash(ingress.recent_fps, 1)],
        ["processed_frame", valueOrDash(lag.latest_result_frame_id)],
        ["frame_id_lag", valueOrDash(lag.frame_id_lag)],
        ["state", valueOrDash(snapshot.state)],
        ["track_id", valueOrDash(snapshot.track_id)],
        ["processing_ms", numberOrDash(result.processing_ms, 2)],
        ["source_to_result_ms", numberOrDash(result.source_to_result_ms, 2)],
        ["ingress_to_runtime_ms", numberOrDash(result.ingress_to_runtime_ms, 2)],
        ["result_age_ms", numberOrDash(lag.latest_result_age_ms, 1)],
        ["frame_age_ms", numberOrDash(lag.latest_frame_age_ms, 1)],
      ];
      summary.innerHTML = items.map(([name, value]) => `<div class="metric"><span>${name}</span><span>${value}</span></div>`).join("");
    }
    function refreshFrame() {
      if (!hasFrame) return;
      frame.src = "/api/frame/latest.jpg?overlay=true&t=" + Date.now();
    }
    async function refreshMetrics() {
      try {
        const response = await fetch("/api/metrics", {cache: "no-store"});
        const data = await response.json();
        hasFrame = Boolean(data.frame);
        frame.style.display = hasFrame ? "block" : "none";
        empty.style.display = hasFrame ? "none" : "block";
        renderSummary(data);
        metrics.textContent = JSON.stringify(data, null, 2);
        if (hasFrame) refreshFrame();
      } catch (error) {
        metrics.textContent = String(error);
      }
    }
    frame.addEventListener("click", async (event) => {
      const rect = frame.getBoundingClientRect();
      const scaleX = frame.naturalWidth / rect.width;
      const scaleY = frame.naturalHeight / rect.height;
      const x = Math.round((event.clientX - rect.left) * scaleX);
      const y = Math.round((event.clientY - rect.top) * scaleY);
      await fetch("/api/commands/click", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({x, y})
      });
      refreshMetrics();
    });
    document.getElementById("reset").addEventListener("click", async () => {
      await fetch("/api/commands/reset", {method: "POST"});
      refreshMetrics();
    });
    setInterval(refreshFrame, 120);
    setInterval(refreshMetrics, 500);
    refreshFrame();
    refreshMetrics();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
