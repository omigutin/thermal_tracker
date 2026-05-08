"""Серверная запись web-просмотра в видео и JSONL."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from thermal_tracker.core.config import PROJECT_ROOT


@dataclass(frozen=True)
class ContentRect:
    """Область реального изображения внутри RAW-буфера."""

    x: int
    y: int
    width: int
    height: int

    def clamp(self, frame_shape: tuple[int, int] | tuple[int, int, int]) -> "ContentRect":
        frame_h, frame_w = frame_shape[:2]
        x = min(max(0, self.x), max(0, frame_w - 1))
        y = min(max(0, self.y), max(0, frame_h - 1))
        width = min(max(1, self.width), frame_w - x)
        height = min(max(1, self.height), frame_h - y)
        return ContentRect(x=x, y=y, width=width, height=height)

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class RecordingFrameMetadata:
    """Метаданные кадра, которые кладём рядом с видео."""

    frame: dict[str, Any]
    result: dict[str, Any] | None
    content_rect: ContentRect | None
    source: dict[str, Any]


class WebRecordingSession:
    """Одна активная запись web-клиента."""

    def __init__(
        self,
        *,
        output_dir: Path,
        base_name: str,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_base_name = _safe_stem(base_name)
        self.video_path = self.output_dir / f"{safe_base_name}.mp4"
        self.jsonl_path = self.output_dir / f"{safe_base_name}.jsonl"
        self.frame_size = frame_size
        self.frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.video_path), fourcc, max(float(fps), 1.0), frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Не удалось открыть запись видео: {self.video_path}")
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")

    def write_frame(self, image: np.ndarray, metadata: RecordingFrameMetadata) -> None:
        """Пишет один кадр и JSONL-событие."""

        visible = _crop_image(image, metadata.content_rect)
        bgr = cv2.cvtColor(visible, cv2.COLOR_GRAY2BGR) if visible.ndim == 2 else visible
        if (bgr.shape[1], bgr.shape[0]) != self.frame_size:
            bgr = cv2.resize(bgr, self.frame_size, interpolation=cv2.INTER_AREA)
        self._writer.write(bgr)
        self.frame_count += 1
        self._jsonl.write(
            json.dumps(
                {
                    "event": "frame",
                    "recorded_frame": self.frame_count,
                    "video_path": str(self.video_path),
                    "jsonl_path": str(self.jsonl_path),
                    "content_rect": metadata.content_rect.as_dict() if metadata.content_rect else None,
                    "source": metadata.source,
                    "frame": metadata.frame,
                    "result": metadata.result,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._jsonl.flush()

    def close(self) -> dict[str, Any]:
        """Закрывает файлы записи и возвращает итог."""

        self._writer.release()
        self._jsonl.close()
        return {
            "video_path": str(self.video_path),
            "jsonl_path": str(self.jsonl_path),
            "frames": self.frame_count,
        }


class WebRecordingManager:
    """Управляет активной записью web-клиента."""

    def __init__(self, *, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or PROJECT_ROOT / "out"
        self._session: WebRecordingSession | None = None

    @property
    def active(self) -> bool:
        return self._session is not None

    def start(self, *, base_name: str, fps: float, frame_size: tuple[int, int]) -> dict[str, Any]:
        """Запускает новую запись, предварительно закрывая старую."""

        if self._session is not None:
            self.stop()
        self._session = WebRecordingSession(
            output_dir=self.output_dir,
            base_name=base_name,
            fps=fps,
            frame_size=frame_size,
        )
        return self.status()

    def write_frame(self, image: np.ndarray, metadata: RecordingFrameMetadata) -> None:
        """Пишет кадр, если запись сейчас включена."""

        if self._session is None:
            return
        self._session.write_frame(image, metadata)

    def stop(self) -> dict[str, Any]:
        """Останавливает активную запись."""

        if self._session is None:
            return {"active": False}
        session = self._session
        self._session = None
        return {"active": False, **session.close()}

    def status(self) -> dict[str, Any]:
        """Возвращает состояние записи."""

        if self._session is None:
            return {"active": False}
        return {
            "active": True,
            "video_path": str(self._session.video_path),
            "jsonl_path": str(self._session.jsonl_path),
            "frames": self._session.frame_count,
        }


def parse_content_rect(data: dict[str, Any] | None) -> ContentRect | None:
    """Разбирает content_rect из JSON или query-параметров."""

    if not isinstance(data, dict):
        return None
    try:
        return ContentRect(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _crop_image(image: np.ndarray, rect: ContentRect | None) -> np.ndarray:
    if rect is None:
        return image
    clean_rect = rect.clamp(image.shape)
    return image[
        clean_rect.y : clean_rect.y + clean_rect.height,
        clean_rect.x : clean_rect.x + clean_rect.width,
    ]


def _safe_stem(raw_name: str) -> str:
    clean = Path(raw_name or "").stem.strip()
    if not clean:
        clean = "thermal_tracker_recording"
    clean = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", clean)
    return clean.strip("._-") or "thermal_tracker_recording"
