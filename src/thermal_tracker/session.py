"""Сессионный контроллер для интерактивного трекинга."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TrackerPreset, build_preset
from .domain.models import DetectedObject, ProcessedFrame, TrackSnapshot
from .domain.runtime import SessionRuntimeState
from .scenarios import ScenarioFactory
from .connections.frames import OpenCvVideoSource


@dataclass(frozen=True)
class LaunchOptions:
    """Параметры, с которыми открывается одна рабочая сессия."""
    video_path: str
    preset_name: str = "opencv_general"
    delay_ms: int = 30
    neural_model_path: str = ""
    neural_tracker_config_path: str = ""

    @property
    def safe_delay_ms(self) -> int:
        """Возвращает безопасную задержку, чтобы `waitKey` не получил ноль."""
        return max(1, int(self.delay_ms))


class TrackingSession:
    """Сессия, собранная вокруг pipeline и источника кадров."""

    def __init__(self, options: LaunchOptions) -> None:
        self.options = options
        self.runtime = SessionRuntimeState(paused=True)
        self.video_source = OpenCvVideoSource(options.video_path)
        self.pipeline = self._build_pipeline(options)
        self.finished = False
        self._render_revision = 0

    @property
    def preset(self) -> TrackerPreset:
        """Возвращает активный пресет целиком."""
        return self.pipeline.preset

    @property
    def preset_name(self) -> str:
        """Короткое имя текущего пресета."""
        return self.pipeline.preset_name

    @property
    def current_frame(self) -> ProcessedFrame | None:
        """Последний обработанный кадр."""
        return self.pipeline.current_frame

    @property
    def current_snapshot(self) -> TrackSnapshot:
        """Последний снимок состояния трекера."""
        return self.pipeline.current_snapshot

    @property
    def candidate_objects(self) -> tuple[DetectedObject, ...]:
        """Текущие кандидаты для GUI, если активный pipeline умеет их отдавать."""

        return tuple(getattr(self.pipeline, "candidate_objects", ()))

    @property
    def pending_click(self) -> tuple[int, int] | None:
        """Последний ещё не обработанный клик."""
        return self.runtime.pending_click

    @property
    def paused(self) -> bool:
        """Признак паузы воспроизведения."""
        return self.runtime.paused

    @property
    def safe_delay_ms(self) -> int:
        """Безопасная задержка между кадрами."""
        return self.options.safe_delay_ms

    @property
    def fps(self) -> float:
        """Частота кадров активного источника видео."""

        return self.video_source.fps

    @property
    def frame_count(self) -> int:
        """Общее число кадров активного видео."""

        return self.video_source.frame_count

    @property
    def current_frame_index(self) -> int:
        """Индекс текущего отображаемого кадра."""

        return self.video_source.current_frame_index

    @property
    def duration_seconds(self) -> float:
        """Полная длительность ролика в секундах."""

        if self.frame_count <= 0:
            return 0.0
        return self.frame_count / max(self.fps, 1e-6)

    @property
    def render_revision(self) -> int:
        """Версия визуального состояния, чтобы не записывать один и тот же кадр бесконечно."""

        return self._render_revision

    def request_click(self, point: tuple[int, int]) -> None:
        """Ставит в очередь клик по цели."""
        self.runtime.pending_click = point
        if self.runtime.paused and self.current_frame is not None:
            self.pipeline.apply_static_actions(self.runtime)
            self._render_revision += 1

    def request_reset(self) -> None:
        """Просит сбросить текущий трек."""
        self.runtime.reset_requested = True
        if self.runtime.paused and self.current_frame is not None:
            self.pipeline.apply_static_actions(self.runtime)
            self._render_revision += 1

    def toggle_pause(self) -> None:
        """Переключает паузу воспроизведения."""
        self.runtime.paused = not self.runtime.paused

    def request_step(self) -> None:
        """Разрешает сделать ровно один следующий шаг по кадрам."""
        self.runtime.paused = True
        self.runtime.step_once = True

    def seek_to_frame(self, frame_index: int, *, keep_pause_state: bool = True) -> None:
        """Переоткрывает сессию и переходит к нужному кадру.

        Такой путь чуть тяжелее прямого `capture.set`, зато сохраняет внутренности
        pipeline в честном состоянии и не оставляет стабилизацию с трекером в шоке.
        """

        target_frame = max(0, int(frame_index))
        total_frames = self.frame_count
        if total_frames > 0:
            target_frame = min(target_frame, total_frames - 1)

        paused_state = self.runtime.paused if keep_pause_state else True
        self.video_source.close()
        self.runtime = SessionRuntimeState(paused=paused_state)
        self.video_source = OpenCvVideoSource(self.options.video_path)
        self.pipeline = self._build_pipeline(self.options)
        self.finished = False
        self.video_source.seek_frame(target_frame)
        self.advance()
        self._render_revision += 1

    def advance(self) -> None:
        """Продвигает сессию вперёд настолько, насколько это сейчас нужно."""
        if self.finished:
            return

        should_advance = not self.runtime.paused or self.runtime.step_once or self.current_frame is None
        if should_advance:
            ok, raw_frame = self.video_source.read()
            self.runtime.step_once = False
            if not ok or raw_frame is None:
                self.finished = True
                return

            self.pipeline.process_next_raw_frame(raw_frame, self.runtime)
            self._render_revision += 1
            return

        if self.current_frame is not None and (self.runtime.pending_click is not None or self.runtime.reset_requested):
            self.pipeline.apply_static_actions(self.runtime)
            self._render_revision += 1

    def close(self) -> None:
        """Освобождает ресурсы сессии."""
        self.video_source.close()

    @staticmethod
    def _build_pipeline(options: LaunchOptions):
        """Создаёт нужный pipeline по типу, который описан в пресете."""

        preset = build_preset(options.preset_name)
        if preset.neural is not None:
            if options.neural_model_path.strip():
                preset.neural.model_path = options.neural_model_path.strip()
            if options.neural_tracker_config_path.strip():
                preset.neural.tracker_config_path = options.neural_tracker_config_path.strip()
        return ScenarioFactory.create_from_preset(preset)
