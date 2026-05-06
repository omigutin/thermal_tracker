"""HTTP-gateway между сетью, браузером и локальной Shared Memory.

Gateway живёт на той же машине, что и Shared Memory. Внешние устройства
говорят с ним по HTTP, а runtime продолжает читать и писать локальные буферы.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

    def publish_raw_y8(
        self,
        payload: bytes,
        *,
        frame_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> dict[str, Any]:
        """Публикует сетевой RAW Y8 кадр в Shared Memory."""

        expected_size = self.config.width * self.config.height
        if len(payload) != expected_size:
            raise ValueError(f"RAW Y8 кадр должен занимать {expected_size} байт, получено {len(payload)}.")

        image = np.frombuffer(payload, dtype=np.uint8).reshape((self.config.height, self.config.width))
        written = self.frame_buffer.write_frame(
            image,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns or now_ns(),
        )
        return _frame_metadata_to_dict(written)

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
        return {
            "prefix": self.config.prefix,
            "camera_id": self.config.camera_id,
            "width": self.config.width,
            "height": self.config.height,
            "frame": _frame_metadata_to_dict(frame) if frame is not None else None,
            "result": result,
        }

    def close(self) -> None:
        """Закрывает handle-ы Shared Memory."""

        self.frame_buffer.close()
        self.command_buffer.close()
        if self._result_buffer is not None:
            self._result_buffer.close()
            self._result_buffer = None

    def _ensure_result_buffer(self) -> SharedMemoryJsonBuffer | None:
        """Лениво подключается к результатам, которые создаёт runtime."""

        if self._result_buffer is not None:
            return self._result_buffer
        self._result_buffer = SharedMemoryJsonBuffer.try_open(prefix=self.config.prefix, kind="results")
        return self._result_buffer


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
            metadata = gateway.publish_raw_y8(payload, frame_id=frame_id, timestamp_ns=timestamp_ns)
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
    <pre id="metrics">Ожидание кадров...</pre>
  </aside>
  <script>
    const frame = document.getElementById("frame");
    const empty = document.getElementById("empty");
    const metrics = document.getElementById("metrics");
    let hasFrame = false;
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
