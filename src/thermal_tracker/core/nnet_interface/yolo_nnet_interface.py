"""Интерфейс к Ultralytics YOLO в режиме track().

Используем его как временную реализацию до подключения общей библиотеки:
- модель принимает кадр;
- внешний tracker (ByteTrack/BoT-SORT) выдаёт track id;
- на выходе получаем список `CandidateFormerResult`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ..config import NeuralConfig, PROJECT_ROOT
from ..domain.models import BoundingBox
from ..stages.candidate_formation.result import CandidateFormerResult
from .base_nnet_interface import BaseNnetInterface


def _resolve_path(raw_path: str) -> str:
    """Приводит путь к абсолютному относительно корня проекта."""

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _ensure_three_channel_frame(frame: np.ndarray) -> np.ndarray:
    """Приводит кадр к 3-канальному виду, который ожидает YOLO."""

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame


class YoloNnetInterface(BaseNnetInterface):
    """Адаптер к YOLO на базе `ultralytics.YOLO`."""

    interface_name = "yolo"
    is_ready = True

    def __init__(self, config: NeuralConfig) -> None:
        if not config.model_path.strip():
            raise RuntimeError("Для NN-сценария не задан путь к файлу модели.")

        self.config = config
        self.model_path = _resolve_path(config.model_path)
        self.tracker_config_path = _resolve_path(config.tracker_config_path) if config.tracker_config_path.strip() else ""
        self.model = YOLO(self.model_path)
        self._track_mode_enabled = self._check_track_mode_available()
        self.mode_name = "track" if self._track_mode_enabled else "predict"

    def track(self, frame: np.ndarray) -> list[CandidateFormerResult]:
        """Запускает модель и возвращает детекции в доменных структурах."""

        model_frame = _ensure_three_channel_frame(frame)
        results = self._run_model(model_frame)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=float)
        class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.full((xyxy.shape[0],), -1, dtype=int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full((xyxy.shape[0],), -1, dtype=int)
        names = result.names or {}

        detections: list[CandidateFormerResult] = []
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            bbox = BoundingBox(
                x=min(x1, x2),
                y=min(y1, y2),
                width=max(1, abs(x2 - x1)),
                height=max(1, abs(y2 - y1)),
            ).clamp(model_frame.shape)
            class_id = int(class_ids[index]) if index < len(class_ids) else -1
            track_id = (
                int(track_ids[index])
                if self._track_mode_enabled and index < len(track_ids) and track_ids[index] >= 0
                else None
            )
            label = str(names.get(class_id, f"class_{class_id}"))
            detections.append(
                CandidateFormerResult(
                    bbox=bbox,
                    area=bbox.area,
                    confidence=float(confs[index]) if index < len(confs) else 0.0,
                    label=label,
                    source=f"{self.interface_name}_{self.mode_name}",
                    track_id=track_id,
                    class_id=class_id if class_id >= 0 else None,
                )
            )

        return detections

    def _check_track_mode_available(self) -> bool:
        """Проверяет, можно ли использовать внешний tracker.

        Ultralytics для ByteTrack/BotSort требует пакет `lap`. Если его нет,
        честно откатываемся на `predict()` и не запускаем автоустановку.
        """

        if not self.tracker_config_path:
            return False
        try:
            import lap  # noqa: F401
        except Exception:
            return False
        return True

    def _build_common_kwargs(self) -> dict[str, object]:
        """Собирает общие аргументы для `predict()` и `track()`."""

        kwargs: dict[str, object] = {
            "verbose": False,
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
        }
        if self.config.device.strip():
            kwargs["device"] = self.config.device.strip()
        if self.config.allowed_classes:
            kwargs["classes"] = list(self.config.allowed_classes)
        return kwargs

    def _run_model(self, frame: np.ndarray):
        """Запускает модель в режиме `track()` или безопасно откатывается к `predict()`."""

        kwargs = self._build_common_kwargs()
        if self._track_mode_enabled:
            kwargs["persist"] = True
            kwargs["tracker"] = self.tracker_config_path
            return self.model.track(frame, **kwargs)
        return self.model.predict(frame, **kwargs)
