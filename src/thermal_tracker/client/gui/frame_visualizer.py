"""Отрисовка рамок и формирование текстового статуса для GUI.

Этот модуль живёт в `gui`, потому что он не участвует в поиске цели,
не принимает алгоритмических решений и не должен считаться стадией пайплайна.
Его задача простая: красиво показать текущее состояние человеку.
"""

from __future__ import annotations

import cv2
import numpy as np

from thermal_tracker.core.config import VisualizationConfig
from thermal_tracker.core.domain.models import BoundingBox, ProcessedFrame, TrackerState
from thermal_tracker.core.stages.target_tracking.result import TargetTrackingResult
from thermal_tracker.core.stages.candidate_formation.result import CandidateFormerResult


def _draw_box(
    image: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
    *,
    font_scale: float = 0.5,
    text_thickness: int = 1,
) -> None:
    x, y, w, h = bbox.to_xywh()
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )


def _format_state_name(state: TrackerState) -> str:
    """Делает название состояния понятнее для человека."""

    if state == TrackerState.IDLE:
        return "Ожидание"
    if state == TrackerState.TRACKING:
        return "Сопровождение"
    if state == TrackerState.SEARCHING:
        return "Повторный поиск"
    if state == TrackerState.RECOVERING:
        return "Восстановление"
    if state == TrackerState.LOST:
        return "Цель потеряна"
    return state.value


def _build_candidate_label(detection: CandidateFormerResult) -> str:
    """Собирает короткую подпись для нейросетевого кандидата."""

    if detection.track_id is not None:
        return f"#{detection.track_id}"
    return ""


def build_status_lines(
    snapshot: TargetTrackingResult,
    preset_name: str,
    *,
    paused: bool = False,
    finished: bool = False,
) -> list[str]:
    """Собирает текстовые строки со служебной информацией."""

    lines = [
        f"Пресет: {preset_name}",
        f"Состояние: {_format_state_name(snapshot.state)}",
        f"Трек ID: {snapshot.track_id if snapshot.track_id is not None else '-'}",
        f"Оценка совпадения: {snapshot.score:.2f}",
        f"Пропущено кадров: {snapshot.lost_frames}",
        f"Режим воспроизведения: {'Пауза' if paused else 'Воспроизведение'}",
        "Управление: ЛКМ выбрать цель, ПКМ сбросить",
        "Клавиши: пробел пауза, N шаг, R сброс, Q/Esc выход",
    ]

    if snapshot.global_motion.valid:
        lines.append(
            "Сдвиг камеры: "
            f"dx={snapshot.global_motion.dx:.1f} "
            f"dy={snapshot.global_motion.dy:.1f} "
            f"resp={snapshot.global_motion.response:.2f}"
        )
    elif snapshot.global_motion.response > 0:
        lines.append(f"Сдвиг камеры: ненадёжно, resp={snapshot.global_motion.response:.2f}")

    if snapshot.message:
        lines.append(f"Сообщение: {snapshot.message}")

    if finished:
        lines.append("Видео завершено. Можно изучить последний кадр или выбрать другой файл.")

    return lines


def build_status_text(
    snapshot: TargetTrackingResult,
    preset_name: str,
    *,
    paused: bool = False,
    finished: bool = False,
) -> str:
    """Собирает готовый текст для нижней панели GUI."""

    return "\n".join(build_status_lines(snapshot, preset_name, paused=paused, finished=finished))


def render_frame(
    frame: ProcessedFrame,
    snapshot: TargetTrackingResult,
    visualization: VisualizationConfig,
    preset_name: str,
    pending_click: tuple[int, int] | None = None,
    candidate_objects: tuple[CandidateFormerResult, ...] | list[CandidateFormerResult] | None = None,
    *,
    include_status_bar: bool = True,
) -> np.ndarray:
    """Рисует рамки, подсказки и при необходимости старую OpenCV-панель."""

    canvas = frame.bgr.copy()
    thickness = max(1, min(2, visualization.line_thickness))

    if visualization.show_search_region and snapshot.search_region is not None and snapshot.state == TrackerState.SEARCHING:
        _draw_box(canvas, snapshot.search_region, (0, 165, 255), 1)

    if candidate_objects and snapshot.state in {TrackerState.IDLE, TrackerState.SEARCHING}:
        show_candidate_labels = snapshot.state == TrackerState.IDLE
        for detection in candidate_objects:
            _draw_box(
                canvas,
                detection.bbox,
                (255, 255, 0),
                1,
                _build_candidate_label(detection) if show_candidate_labels else None,
                font_scale=0.46,
                text_thickness=1,
            )

    if visualization.show_predicted_box and snapshot.predicted_bbox is not None and snapshot.state == TrackerState.SEARCHING:
        _draw_box(canvas, snapshot.predicted_bbox, (255, 0, 255), 1)

    if snapshot.bbox is not None:
        label = f"#{snapshot.track_id}" if snapshot.track_id is not None else ""
        _draw_box(canvas, snapshot.bbox, (0, 255, 0), thickness, label, font_scale=0.38, text_thickness=1)

    if pending_click is not None:
        cv2.circle(canvas, pending_click, 6, (0, 0, 255), -1)
        cv2.circle(canvas, pending_click, 14, (0, 255, 255), 1)

    if include_status_bar:
        _draw_status_bar(canvas, snapshot, preset_name)
    return canvas


def _draw_status_bar(image: np.ndarray, snapshot: TargetTrackingResult, preset_name: str) -> None:
    """Рисует старую OpenCV-панель состояния поверх кадра."""

    lines = build_status_lines(snapshot, preset_name)

    panel_height = 24 + len(lines) * 22
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (min(image.shape[1] - 8, 560), min(image.shape[0] - 8, panel_height)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0, dst=image)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (16, 30 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
