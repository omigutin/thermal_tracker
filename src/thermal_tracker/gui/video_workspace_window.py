"""Главное окно нового GUI в стиле простого видеоплеера.

Слева живут:
- выбор каталога с видео;
- список найденных роликов;
- пресет и задержка воспроизведения;
- кнопки управления текущей сессией.

Справа живёт чистое видео без служебного текста поверх цели.
Снизу находится копируемая техническая панель.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from ..config import AVAILABLE_PRESETS, PROJECT_ROOT, build_preset, get_preset_presentation
from ..errors import VideoOpenError
from ..session import LaunchOptions, TrackingSession
from .frame_visualizer import build_status_text, render_frame
from .tooltips import HoverTooltip

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}
SOURCE_KIND_LABELS = {
    "video_directory": "Каталог с видео",
    "video_files": "Один или несколько видеофайлов",
    "shared_memory": "Shared Memory (скоро)",
}
VIDEO_DIALOG_TYPES = [
    ("Видеофайлы", "*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg *.wmv *.m4v"),
    ("Все файлы", "*.*"),
]


class TrackingPlayerWindow:
    """Единое окно для выбора видео, запуска и ручного трекинга."""

    def __init__(
        self,
        *,
        default_video: str = "",
        default_preset: str = "opencv_general",
        default_delay_ms: int = 30,
        auto_start: bool = False,
    ) -> None:
        self.session: TrackingSession | None = None
        self.root = tk.Tk()
        self.root.title("Тепловизионный трекер")
        self.root.geometry("1720x980")
        self.root.minsize(1320, 820)

        self._closed = False
        self._tick_after_id: str | None = None
        self._photo: tk.PhotoImage | None = None
        self._last_rendered_frame = None
        self._preview_video_path: Path | None = None
        self._preview_frame = None
        self._record_output_path: Path | None = None
        self._video_writer: cv2.VideoWriter | None = None
        self._last_written_render_revision = -1
        self._display_box = (0, 0, 1, 1, 1, 1)

        self._source_label_to_kind = {label: kind for kind, label in SOURCE_KIND_LABELS.items()}
        self._preset_name_to_display = self._build_preset_display_map()
        self._preset_display_to_name = {display: name for name, display in self._preset_name_to_display.items()}
        default_preset_name = default_preset if default_preset in AVAILABLE_PRESETS else AVAILABLE_PRESETS[0]

        self._directory_var = tk.StringVar()
        self._source_kind_var = tk.StringVar(value=SOURCE_KIND_LABELS["video_directory"])
        self._preset_var = tk.StringVar(value=self._preset_name_to_display[default_preset_name])
        self._delay_var = tk.StringVar(value=str(default_delay_ms))
        self._record_video_var = tk.BooleanVar(value=False)
        self._preset_summary_var = tk.StringVar()
        self._playlist_status_var = tk.StringVar(value="Источник пока не выбран.")
        self._model_var = tk.StringVar()
        self._tracker_var = tk.StringVar()
        self._timeline_var = tk.DoubleVar(value=0.0)
        self._time_label_var = tk.StringVar(value="00:00 / 00:00")
        self._speed_factor_var = tk.StringVar(value="1.0")

        self._video_files: list[Path] = []
        self._selected_video_path: Path | None = None
        self._timeline_dragging = False
        self._cached_model_choices: list[str] | None = None
        self._cached_tracker_choices: list[str] | None = None

        self._reset_button: ttk.Button | None = None
        self._info_text: tk.Text | None = None
        self._video_canvas: tk.Canvas | None = None
        self._video_listbox: tk.Listbox | None = None
        self._source_browse_button: ttk.Button | None = None
        self._source_kind_box: ttk.Combobox | None = None
        self._preset_box: ttk.Combobox | None = None
        self._model_box: ttk.Combobox | None = None
        self._tracker_box: ttk.Combobox | None = None
        self._timeline_scale: ttk.Scale | None = None
        self._speed_entry: ttk.Entry | None = None
        self._play_pause_button: ttk.Button | None = None
        self._back_frame_button: ttk.Button | None = None
        self._forward_frame_button: ttk.Button | None = None
        self._rewind_button: ttk.Button | None = None
        self._forward_button: ttk.Button | None = None
        self._restart_button: ttk.Button | None = None
        self._stop_playback_button: ttk.Button | None = None
        self._slower_button: ttk.Button | None = None
        self._faster_button: ttk.Button | None = None

        self._build_layout()
        self._bind_events()
        self._update_preset_info()
        self._on_source_kind_changed()
        self._refresh_view()
        self.root.after_idle(lambda: self._finish_startup(default_video, auto_start))

    def run(self) -> None:
        """Запускает цикл событий Tk."""

        self.root.mainloop()

    def _build_preset_display_map(self) -> dict[str, str]:
        """Собирает человекочитаемые названия пресетов для комбобокса."""

        display_map: dict[str, str] = {}
        used_titles: set[str] = set()
        for preset_name in AVAILABLE_PRESETS:
            title = get_preset_presentation(preset_name).title.strip() or preset_name
            display = title if title not in used_titles else f"{title} [{preset_name}]"
            display_map[preset_name] = display
            used_titles.add(display)
        return display_map

    def _selected_source_kind(self) -> str:
        """Возвращает технический код выбранного типа источника."""

        return self._source_label_to_kind.get(self._source_kind_var.get(), "video_directory")

    def _selected_preset_name(self) -> str:
        """Возвращает техническое имя выбранного пресета."""

        return self._preset_display_to_name.get(self._preset_var.get(), AVAILABLE_PRESETS[0])

    def _finish_startup(self, default_video: str, auto_start: bool) -> None:
        """Догружает плейлист и превью уже после появления окна."""

        if self._closed:
            return
        self._load_initial_directory(default_video)
        self._refresh_view()
        if auto_start and self._selected_video_path is not None:
            self._open_selected_video()

    @staticmethod
    def _scan_relative_files(directory: Path, suffixes: set[str]) -> list[str]:
        """Собирает относительные пути файлов из каталога, если он существует."""

        if not directory.exists():
            return []
        return sorted(
            [
                str(path.relative_to(PROJECT_ROOT))
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            ],
            key=str.lower,
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Преобразует секунды в человекочитаемое время для панели плеера."""

        safe_seconds = max(0, int(round(seconds)))
        minutes, secs = divmod(safe_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _clamp_speed_factor(value: float) -> float:
        """Не даёт скорости уйти в ноль или в космос."""

        return min(16.0, max(0.05, float(value)))

    def _get_speed_factor(self) -> float:
        """Читает текущую скорость из поля ввода и мягко чинит мусор."""

        try:
            raw_value = float(self._speed_factor_var.get().strip().replace(",", "."))
        except ValueError:
            raw_value = 1.0
        value = self._clamp_speed_factor(raw_value)
        self._speed_factor_var.set(f"{value:.2f}".rstrip("0").rstrip("."))
        return value

    def _set_speed_factor(self, value: float) -> None:
        """Устанавливает новую скорость в удобном человекочитаемом виде."""

        clamped = self._clamp_speed_factor(value)
        self._speed_factor_var.set(f"{clamped:.2f}".rstrip("0").rstrip("."))

    def _apply_speed_factor_entry(self) -> None:
        """Нормализует ручной ввод скорости и при необходимости будит плеер."""

        self._set_speed_factor(self._get_speed_factor())
        self._refresh_view()
        if self.session is not None and not self.session.finished and not self.session.paused:
            self._schedule_tick(0)

    def _get_model_choices(self) -> list[str]:
        """Лениво собирает список доступных файлов моделей и кэширует его."""

        if self._cached_model_choices is None:
            self._cached_model_choices = self._scan_relative_files(
                PROJECT_ROOT / "models",
                {".pt", ".onnx", ".engine", ".bin"},
            )
        return list(self._cached_model_choices)

    def _get_tracker_choices(self) -> list[str]:
        """Лениво собирает список tracker-конфигов и кэширует его."""

        if self._cached_tracker_choices is None:
            self._cached_tracker_choices = self._scan_relative_files(
                PROJECT_ROOT / "trackers",
                {".yaml", ".yml", ".json", ".toml"},
            )
        return list(self._cached_tracker_choices)

    def _build_layout(self) -> None:
        """Собирает главное окно."""

        self.root.columnconfigure(0, weight=0, minsize=440)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        style = ttk.Style(self.root)
        style.configure("Player.TButton", padding=(10, 7), font=("Segoe UI Symbol", 13))

        left_frame = ttk.Frame(self.root, padding=(12, 12, 10, 8))
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)

        right_frame = ttk.Frame(self.root, padding=(0, 12, 12, 8))
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=0, minsize=360)
        right_frame.rowconfigure(0, weight=1)

        source_frame = ttk.LabelFrame(left_frame, text="Источник", padding=10)
        source_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        source_frame.columnconfigure(1, weight=1)

        ttk.Label(source_frame, text="Тип:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        self._source_kind_box = ttk.Combobox(
            source_frame,
            textvariable=self._source_kind_var,
            values=list(SOURCE_KIND_LABELS.values()),
            state="readonly",
        )
        self._source_kind_box.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        self._source_kind_box.bind("<<ComboboxSelected>>", lambda _event: self._on_source_kind_changed())

        ttk.Label(source_frame, text="Путь:").grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(source_frame, textvariable=self._directory_var).grid(row=1, column=1, sticky="ew", padx=(0, 8))
        self._source_browse_button = ttk.Button(source_frame, text="Выбрать...", command=self._browse_directory)
        self._source_browse_button.grid(row=1, column=2, sticky="ew")
        ttk.Label(
            source_frame,
            textvariable=self._playlist_status_var,
            justify="left",
            foreground="#4f4f4f",
            wraplength=390,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        settings_frame = ttk.LabelFrame(left_frame, text="Режим", padding=10)
        settings_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Пресет:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        self._preset_box = ttk.Combobox(
            settings_frame,
            textvariable=self._preset_var,
            values=[self._preset_name_to_display[name] for name in AVAILABLE_PRESETS],
            state="readonly",
        )
        self._preset_box.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        self._preset_box.bind("<<ComboboxSelected>>", lambda _event: self._update_preset_info())
        HoverTooltip(self._preset_box, lambda: get_preset_presentation(self._selected_preset_name()).tooltip)

        ttk.Label(settings_frame, text="Описание:").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=(0, 6))
        ttk.Label(
            settings_frame,
            textvariable=self._preset_summary_var,
            justify="left",
            wraplength=290,
        ).grid(row=1, column=1, sticky="w", pady=(0, 6))

        ttk.Label(settings_frame, text="Модель:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(4, 6))
        self._model_box = ttk.Combobox(settings_frame, textvariable=self._model_var, state="disabled")
        self._model_box.grid(row=2, column=1, sticky="ew", pady=(4, 6))

        ttk.Label(settings_frame, text="Трекер:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(0, 0))
        self._tracker_box = ttk.Combobox(settings_frame, textvariable=self._tracker_var, state="disabled")
        self._tracker_box.grid(row=3, column=1, sticky="ew", pady=(0, 0))

        library_frame = ttk.LabelFrame(left_frame, text="Плейлист", padding=10)
        library_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        library_frame.columnconfigure(0, weight=1)
        library_frame.rowconfigure(0, weight=1)

        self._video_listbox = tk.Listbox(
            library_frame,
            activestyle="dotbox",
            exportselection=False,
            font=("Segoe UI", 11),
            height=14,
        )
        self._video_listbox.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(library_frame, orient="vertical", command=self._video_listbox.yview)
        library_scroll.grid(row=0, column=1, sticky="ns")
        self._video_listbox.configure(yscrollcommand=library_scroll.set)

        self._video_canvas = tk.Canvas(right_frame, background="black", highlightthickness=0, cursor="crosshair")
        self._video_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        controls_frame = ttk.Frame(right_frame, padding=(0, 8, 0, 0))
        controls_frame.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        controls_frame.columnconfigure(0, weight=1)

        timeline_frame = ttk.Frame(controls_frame)
        timeline_frame.grid(row=0, column=0, sticky="ew")
        timeline_frame.columnconfigure(0, weight=1)
        timeline_frame.columnconfigure(1, weight=0)

        self._timeline_scale = ttk.Scale(
            timeline_frame,
            orient="horizontal",
            from_=0.0,
            to=1.0,
            variable=self._timeline_var,
            command=self._on_timeline_changed,
        )
        self._timeline_scale.grid(row=0, column=0, sticky="ew")
        self._timeline_scale.bind("<ButtonPress-1>", self._on_timeline_press)
        self._timeline_scale.bind("<ButtonRelease-1>", self._on_timeline_release)
        ttk.Label(timeline_frame, textvariable=self._time_label_var, width=16, anchor="e").grid(
            row=0,
            column=1,
            sticky="e",
            padx=(10, 0),
        )

        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for column in range(11):
            buttons_frame.columnconfigure(column, weight=0)

        self._restart_button = ttk.Button(buttons_frame, text="⏮", width=4, style="Player.TButton", command=self._restart_video)
        self._restart_button.grid(row=0, column=0, padx=(0, 4))
        self._rewind_button = ttk.Button(
            buttons_frame,
            text="⏪",
            width=4,
            style="Player.TButton",
            command=lambda: self._seek_relative_seconds(-1.0),
        )
        self._rewind_button.grid(row=0, column=1, padx=4)
        self._back_frame_button = ttk.Button(
            buttons_frame,
            text="⏴",
            width=4,
            style="Player.TButton",
            command=lambda: self._seek_relative_frames(-1),
        )
        self._back_frame_button.grid(row=0, column=2, padx=4)
        self._play_pause_button = ttk.Button(
            buttons_frame,
            text="▶",
            width=4,
            style="Player.TButton",
            command=self._toggle_pause,
        )
        self._play_pause_button.grid(row=0, column=3, padx=4)
        self._forward_frame_button = ttk.Button(
            buttons_frame,
            text="⏵",
            width=4,
            style="Player.TButton",
            command=lambda: self._seek_relative_frames(1),
        )
        self._forward_frame_button.grid(row=0, column=4, padx=4)
        self._forward_button = ttk.Button(
            buttons_frame,
            text="⏩",
            width=4,
            style="Player.TButton",
            command=lambda: self._seek_relative_seconds(1.0),
        )
        self._forward_button.grid(row=0, column=5, padx=4)
        self._stop_playback_button = ttk.Button(
            buttons_frame,
            text="⏹",
            width=4,
            style="Player.TButton",
            command=self._stop_playback,
        )
        self._stop_playback_button.grid(row=0, column=6, padx=4)
        self._reset_button = ttk.Button(
            buttons_frame,
            text="✖",
            width=4,
            style="Player.TButton",
            command=self._reset_tracker,
        )
        self._reset_button.grid(row=0, column=7, padx=4)
        self._slower_button = ttk.Button(
            buttons_frame,
            text="➖",
            width=4,
            style="Player.TButton",
            command=lambda: self._change_speed(-1),
        )
        self._slower_button.grid(row=0, column=8, padx=(14, 4))
        self._speed_entry = ttk.Entry(buttons_frame, textvariable=self._speed_factor_var, width=6, justify="center")
        self._speed_entry.grid(row=0, column=9, padx=4)
        self._speed_entry.bind("<Return>", lambda _event: self._apply_speed_factor_entry())
        self._speed_entry.bind("<FocusOut>", lambda _event: self._apply_speed_factor_entry())
        self._faster_button = ttk.Button(
            buttons_frame,
            text="➕",
            width=4,
            style="Player.TButton",
            command=lambda: self._change_speed(1),
        )
        self._faster_button.grid(row=0, column=10, padx=(4, 0), sticky="w")

        HoverTooltip(self._restart_button, lambda: "В начало")
        HoverTooltip(self._rewind_button, lambda: "Назад на 1 секунду")
        HoverTooltip(self._back_frame_button, lambda: "Назад на 1 кадр")
        HoverTooltip(self._play_pause_button, lambda: "Пауза или продолжение")
        HoverTooltip(self._forward_frame_button, lambda: "Вперёд на 1 кадр")
        HoverTooltip(self._forward_button, lambda: "Вперёд на 1 секунду")
        HoverTooltip(self._stop_playback_button, lambda: "Стоп: пауза и возврат к первому кадру")
        HoverTooltip(self._reset_button, lambda: "Сбросить текущий трек")
        HoverTooltip(self._slower_button, lambda: "Замедлить воспроизведение")
        HoverTooltip(self._faster_button, lambda: "Ускорить воспроизведение")
        HoverTooltip(self._speed_entry, lambda: "Скорость воспроизведения: например 0.1, 0.5, 1, 2 или 4")

        info_frame = ttk.LabelFrame(right_frame, text="Техническая информация", padding=10)
        info_frame.grid(row=0, column=1, sticky="nsew")
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        self._info_text = tk.Text(
            info_frame,
            wrap="word",
            width=42,
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        self._info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self._info_text.yview)
        info_scroll.grid(row=0, column=1, sticky="ns")
        self._info_text.configure(yscrollcommand=info_scroll.set)

        info_footer_frame = ttk.Frame(right_frame, padding=(0, 8, 0, 0))
        info_footer_frame.grid(row=1, column=1, sticky="ew")
        info_footer_frame.columnconfigure(2, weight=1)

        ttk.Label(info_footer_frame, text="Скорость:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(
            info_footer_frame,
            textvariable=self._speed_factor_var,
            width=8,
            justify="center",
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(info_footer_frame, text="x").grid(row=0, column=2, sticky="w", padx=(4, 0))
        ttk.Checkbutton(
            info_footer_frame,
            text="Записывать видео в out/",
            variable=self._record_video_var,
            command=self._toggle_recording,
        ).grid(row=0, column=3, sticky="e", padx=(12, 0))

    def _bind_events(self) -> None:
        """Подключает обработчики мыши, клавиатуры и списка видео."""

        assert self._video_canvas is not None
        assert self._video_listbox is not None

        self._video_canvas.bind("<Button-1>", self._on_left_click)
        self._video_canvas.bind("<Button-3>", self._on_right_click)
        self._video_canvas.bind("<Configure>", lambda _event: self._refresh_video_only())

        self._video_listbox.bind("<<ListboxSelect>>", self._on_video_selected)
        self._video_listbox.bind("<Double-Button-1>", self._open_selected_video)
        self._video_listbox.bind("<Return>", self._open_selected_video)

        self.root.bind("<space>", lambda _event: self._toggle_pause())
        self.root.bind("<KeyPress-n>", lambda _event: self._step_once())
        self.root.bind("<KeyPress-r>", lambda _event: self._reset_tracker())
        self.root.bind("<KeyPress-q>", lambda _event: self._close())
        self.root.bind("<Escape>", lambda _event: self._close())
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _load_initial_directory(self, default_video: str) -> None:
        """Подхватывает стартовый каталог, чтобы список не был пустым с нуля."""

        if default_video:
            candidate = Path(default_video).expanduser()
            if candidate.exists():
                self._source_kind_var.set(SOURCE_KIND_LABELS["video_directory"])
                self._load_video_directory(candidate.parent, select_path=candidate)
                return

        default_directory = Path.cwd() / "video"
        if default_directory.exists():
            self._source_kind_var.set(SOURCE_KIND_LABELS["video_directory"])
            self._load_video_directory(default_directory)
        else:
            self._on_source_kind_changed()

    def _schedule_tick(self, delay_ms: int) -> None:
        """Планирует следующее обновление без дубликатов."""

        if self._closed:
            return
        if self._tick_after_id is not None:
            self.root.after_cancel(self._tick_after_id)
        self._tick_after_id = self.root.after(delay_ms, self._tick)

    def _tick(self) -> None:
        """Периодически продвигает сессию вперёд."""

        self._tick_after_id = None
        if self._closed or self.session is None:
            return

        self.session.advance()
        self._refresh_view()

        if self.session.finished:
            return

        speed_factor = self._get_speed_factor()
        delay_ms = 35 if self.session.paused else max(1, int(round(self.session.safe_delay_ms / speed_factor)))
        self._schedule_tick(delay_ms)

    def _restart_video(self) -> str:
        """Возвращает текущее видео в самое начало."""

        if self.session is None:
            if self._selected_video_path is not None:
                self._open_selected_video()
            return "break"
        self.session.seek_to_frame(0)
        self._refresh_view()
        self._schedule_tick(0)
        return "break"

    def _seek_relative_frames(self, delta_frames: int) -> str:
        """Смещает текущую позицию на заданное число кадров."""

        if self.session is None or self.session.finished:
            return "break"
        target_frame = self.session.current_frame_index + int(delta_frames)
        self.session.seek_to_frame(target_frame)
        self._refresh_view()
        self._schedule_tick(0)
        return "break"

    def _seek_relative_seconds(self, delta_seconds: float) -> str:
        """Смещает текущую позицию на заданное число секунд."""

        if self.session is None or self.session.finished:
            return "break"
        delta_frames = int(round(delta_seconds * self.session.fps))
        return self._seek_relative_frames(delta_frames)

    def _seek_to_frame(self, frame_index: int) -> None:
        """Переходит к выбранному кадру через безопасный restart-aware seek."""

        if self.session is None or self.session.finished:
            return
        self.session.seek_to_frame(frame_index)
        self._refresh_view()
        self._schedule_tick(0)

    def _on_timeline_press(self, _event: tk.Event) -> None:
        """Помечает начало ручного перетаскивания бегунка."""

        self._timeline_dragging = True

    def _on_timeline_changed(self, value: str) -> None:
        """Показывает время выбранной позиции, пока пользователь тянет бегунок."""

        if self.session is None or not self._timeline_dragging:
            return

        try:
            frame_index = int(round(float(value)))
        except ValueError:
            return

        total_seconds = self.session.duration_seconds
        current_seconds = frame_index / max(self.session.fps, 1e-6)
        self._time_label_var.set(f"{self._format_seconds(current_seconds)} / {self._format_seconds(total_seconds)}")

    def _on_timeline_release(self, _event: tk.Event) -> None:
        """После отпускания бегунка переводит видео к выбранному кадру."""

        self._timeline_dragging = False
        if self.session is None or self.session.finished:
            return
        self._seek_to_frame(int(round(self._timeline_var.get())))

    def _stop_playback(self) -> str:
        """Останавливает воспроизведение как плеер: пауза и возврат к первому кадру."""

        if self.session is None:
            return "break"

        if not self.session.paused:
            self.session.toggle_pause()
        self.session.seek_to_frame(0)
        self._refresh_view()
        self._schedule_tick(0)
        return "break"

    def _change_speed(self, direction: int) -> str:
        """Меняет скорость вдвое вверх или вниз и сразу обновляет интерфейс."""

        current = self._get_speed_factor()
        updated = current * 2.0 if int(direction) > 0 else current / 2.0
        self._set_speed_factor(updated)
        self._refresh_view()
        if self.session is not None and not self.session.finished and not self.session.paused:
            self._schedule_tick(0)
        return "break"

    def _refresh_view(self) -> None:
        """Обновляет текст, видео и состояние кнопок."""

        self._refresh_info_text()
        self._refresh_video_frame()
        self._update_player_panel()

    def _refresh_info_text(self) -> None:
        """Перерисовывает правую техническую панель."""

        assert self._info_text is not None

        preset_title = get_preset_presentation(self._selected_preset_name()).title
        source_title = self._source_kind_var.get().strip()
        if self.session is None:
            selected_preset = build_preset(self._selected_preset_name())
            is_auto_neural = selected_preset.pipeline_kind == "auto_neural_detection"
            directory_text = self._directory_var.get().strip() or "не выбран"
            selected_name = self._selected_video_path.name if self._selected_video_path is not None else "ничего не выбрано"
            usage_lines = [
                "1. Выберите тип источника.",
                "2. Соберите плейлист слева.",
                "3. Откройте ролик двойным щелчком слева или просто кликните по кадру справа.",
                "4. Видео всегда стартует с паузы на первом кадре.",
            ]
            if is_auto_neural:
                usage_lines.extend(
                    [
                        "5. В этом режиме клик по цели не нужен.",
                        "6. Пробел снимает паузу, N делает шаг, R просто обновляет состояние.",
                    ]
                )
            else:
                usage_lines.extend(
                    [
                        "5. ЛКМ по видео выбирает цель, ПКМ сбрасывает трек.",
                        "6. Пробел снимает паузу, N делает шаг, R сбрасывает трек.",
                    ]
                )
            full_text = (
                f"Источник: {source_title}\n"
                f"Путь: {directory_text}\n"
                f"Файлов в плейлисте: {len(self._video_files)}\n"
                f"Выбрано: {selected_name}\n"
                f"Пресет: {preset_title}\n"
                f"Скорость: {self._get_speed_factor():.2f}x\n"
                f"Запись видео: {self._build_recording_status()}\n\n"
                "Как работать:\n"
                + "\n".join(usage_lines)
            )
        else:
            if self.session.preset.pipeline_kind == "auto_neural_detection":
                candidate_count = len(self.session.candidate_objects)
                text = "\n".join(
                    [
                        f"Пресет: {preset_title}",
                        "Состояние: Автоматический поиск всех объектов",
                        f"Найдено объектов: {candidate_count}",
                        f"Оценка лучшего кандидата: {self.session.current_snapshot.score:.2f}",
                        f"Режим воспроизведения: {'Пауза' if self.session.paused else 'Воспроизведение'}",
                        "Управление: клик по цели не нужен",
                        "Клавиши: пробел пауза, N шаг, Q/Esc выход",
                        f"Сообщение: {self.session.current_snapshot.message}",
                    ]
                )
            else:
                text = build_status_text(
                    self.session.current_snapshot,
                    preset_title,
                    paused=self.session.paused,
                    finished=self.session.finished,
                )
            full_text = (
                f"Источник: {source_title}\n"
                f"Файл: {self.session.options.video_path}\n"
                f"Каталог: {Path(self.session.options.video_path).parent}\n\n"
                f"Скорость: {self._get_speed_factor():.2f}x\n"
                f"Запись видео: {self._build_recording_status()}\n\n"
                f"{text}"
            )

        self._info_text.configure(state="normal")
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert("1.0", full_text)

    def _refresh_video_frame(self) -> None:
        """Готовит новый кадр для правой панели."""

        if self.session is None or self.session.current_frame is None:
            self._last_rendered_frame = self._preview_frame.copy() if self._preview_frame is not None else None
            if self._last_rendered_frame is None:
                self._clear_canvas_message()
            else:
                self._refresh_video_only()
            return

        self._last_rendered_frame = render_frame(
            frame=self.session.current_frame,
            snapshot=self.session.current_snapshot,
            visualization=self.session.preset.visualization,
            preset_name=self.session.preset_name,
            pending_click=self.session.pending_click,
            candidate_objects=self.session.candidate_objects,
            include_status_bar=False,
        )
        self._refresh_video_only()
        self._write_record_frame_if_needed()

    def _refresh_video_only(self) -> None:
        """Перерисовывает уже подготовленный кадр с учётом размера окна."""

        assert self._video_canvas is not None

        if self._last_rendered_frame is None:
            self._clear_canvas_message()
            return

        canvas_width = max(2, self._video_canvas.winfo_width())
        canvas_height = max(2, self._video_canvas.winfo_height())
        frame_height, frame_width = self._last_rendered_frame.shape[:2]
        scale = min(canvas_width / max(frame_width, 1), canvas_height / max(frame_height, 1))
        display_width = max(1, int(round(frame_width * scale)))
        display_height = max(1, int(round(frame_height * scale)))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(self._last_rendered_frame, (display_width, display_height), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm_data = f"P6 {display_width} {display_height} 255\n".encode("ascii") + rgb.tobytes()
        self._photo = tk.PhotoImage(data=ppm_data, format="PPM")

        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        self._display_box = (offset_x, offset_y, display_width, display_height, frame_width, frame_height)

        self._video_canvas.delete("all")
        self._video_canvas.create_image(offset_x, offset_y, anchor="nw", image=self._photo)

    def _clear_canvas_message(self) -> None:
        """Показывает понятную заглушку, если видео сейчас не открыто."""

        assert self._video_canvas is not None

        source_kind = self._selected_source_kind()
        if source_kind == "shared_memory":
            message = "Источник Shared Memory пока не подключён"
        elif not self._video_files:
            message = "Соберите плейлист слева"
        elif self._selected_video_path is None:
            message = "Выберите ролик слева"
        else:
            message = "Двойной щелчок по ролику слева или клик по кадру справа"

        self._video_canvas.delete("all")
        canvas_width = max(2, self._video_canvas.winfo_width())
        canvas_height = max(2, self._video_canvas.winfo_height())
        self._video_canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=message,
            fill="#d0d0d0",
            font=("Segoe UI", 16, "bold"),
        )

    def _update_player_panel(self) -> None:
        """Обновляет нижнюю панель плеера: кнопки, время, бегунок и скорость."""

        has_selection = self._selected_video_path is not None
        has_session = self.session is not None
        has_live_session = has_session and not self.session.finished

        current_index = (
            int(round(self._timeline_var.get()))
            if has_session and self._timeline_dragging
            else (self.session.current_frame_index if has_session else 0)
        )
        total_frames = self.session.frame_count if has_session else 0
        max_frame_index = max(0, total_frames - 1)
        current_seconds = (current_index / max(self.session.fps, 1e-6)) if has_session else 0.0
        total_seconds = self.session.duration_seconds if has_session else 0.0

        self._time_label_var.set(f"{self._format_seconds(current_seconds)} / {self._format_seconds(total_seconds)}")

        if self._timeline_scale is not None:
            self._timeline_scale.configure(to=float(max_frame_index if total_frames > 0 else 1))
            if not self._timeline_dragging:
                self._timeline_var.set(float(current_index))
            self._timeline_scale.configure(state=("normal" if has_live_session else "disabled"))

        if self._restart_button is not None:
            self._restart_button.configure(state=("normal" if has_live_session else "disabled"))
        if self._rewind_button is not None:
            self._rewind_button.configure(state=("normal" if has_live_session else "disabled"))
        if self._back_frame_button is not None:
            self._back_frame_button.configure(state=("normal" if has_live_session else "disabled"))
        if self._play_pause_button is not None:
            self._play_pause_button.configure(
                state=("normal" if has_live_session else "disabled"),
                text=("▶" if self.session is None or self.session.paused else "⏸"),
            )
        if self._forward_frame_button is not None:
            self._forward_frame_button.configure(state=("normal" if has_live_session else "disabled"))
        if self._forward_button is not None:
            self._forward_button.configure(state=("normal" if has_live_session else "disabled"))
        if self._stop_playback_button is not None:
            self._stop_playback_button.configure(state=("normal" if has_session else "disabled"))
        if self._reset_button is not None:
            self._reset_button.configure(state=("normal" if has_session else "disabled"))
        if self._slower_button is not None:
            self._slower_button.configure(state=("normal" if self._get_speed_factor() > 0.05 else "disabled"))
        if self._faster_button is not None:
            self._faster_button.configure(state=("normal" if self._get_speed_factor() < 16.0 else "disabled"))

    def _update_preset_info(self) -> None:
        """Обновляет человеческое описание выбранного пресета."""

        preset_name = self._selected_preset_name()
        info = get_preset_presentation(preset_name)
        combined_description = info.description.strip()
        if info.tooltip.strip() and info.tooltip.strip() not in combined_description:
            combined_description = f"{info.tooltip.strip()}\n\n{combined_description}"
        self._preset_summary_var.set(combined_description)

        preset = build_preset(preset_name)
        is_neural = preset.neural is not None
        if is_neural and preset.neural is not None:
            model_choices = self._get_model_choices()
            tracker_choices = self._get_tracker_choices()
            if preset.neural.model_path and preset.neural.model_path not in model_choices:
                model_choices.insert(0, preset.neural.model_path)
            if preset.neural.tracker_config_path and preset.neural.tracker_config_path not in tracker_choices:
                tracker_choices.insert(0, preset.neural.tracker_config_path)

            if self._model_box is not None:
                self._model_box.configure(values=model_choices, state="readonly")
            if self._tracker_box is not None:
                self._tracker_box.configure(values=tracker_choices, state="readonly")

            if self._model_var.get().strip() not in model_choices:
                self._model_var.set(preset.neural.model_path if preset.neural.model_path else (model_choices[0] if model_choices else ""))
            if self._tracker_var.get().strip() not in tracker_choices:
                self._tracker_var.set(
                    preset.neural.tracker_config_path
                    if preset.neural.tracker_config_path
                    else (tracker_choices[0] if tracker_choices else "")
                )
            return

        self._model_var.set("Недоступно для этого пресета")
        self._tracker_var.set("Недоступно для этого пресета")
        if self._model_box is not None:
            self._model_box.configure(values=(), state="disabled")
        if self._tracker_box is not None:
            self._tracker_box.configure(values=(), state="disabled")

    def _on_source_kind_changed(self) -> None:
        """Переключает режим источника и подготавливает соответствующий UX."""

        source_kind = self._selected_source_kind()
        if self._source_browse_button is not None:
            browse_text = "Папка..." if source_kind == "video_directory" else "Файлы..."
            if source_kind == "shared_memory":
                browse_text = "Пока недоступно"
            self._source_browse_button.configure(
                text=browse_text,
                state=("disabled" if source_kind == "shared_memory" else "normal"),
            )

        self._close_session_only()
        self._video_files = []
        self._selected_video_path = None
        self._preview_video_path = None
        self._preview_frame = None
        assert self._video_listbox is not None
        self._video_listbox.delete(0, tk.END)

        if source_kind == "shared_memory":
            self._directory_var.set("Shared Memory будет подключён позже")
            self._playlist_status_var.set("Для Shared Memory плейлист пока не используется.")
            self._refresh_view()
            return

        self._directory_var.set("")
        self._playlist_status_var.set(
            "Выберите папку с видео." if source_kind == "video_directory" else "Выберите один или несколько видеофайлов."
        )
        self._refresh_view()

    def _browse_directory(self) -> None:
        """Открывает системный выбор папки или файлов в зависимости от источника."""

        source_kind = self._selected_source_kind()
        initial_directory = self._directory_var.get().strip() or str(Path.cwd())

        if source_kind == "video_files":
            chosen = filedialog.askopenfilenames(
                parent=self.root,
                title="Выбор одного или нескольких видеофайлов",
                initialdir=initial_directory if Path(initial_directory).exists() else str(Path.cwd()),
                filetypes=VIDEO_DIALOG_TYPES,
            )
            if chosen:
                self._load_video_files([Path(path) for path in chosen])
            return

        if source_kind == "shared_memory":
            return

        chosen = filedialog.askdirectory(
            parent=self.root,
            title="Выбор каталога с видео",
            initialdir=initial_directory if Path(initial_directory).exists() else str(Path.cwd()),
        )
        if chosen:
            self._load_video_directory(Path(chosen))

    def _apply_video_playlist(
        self,
        video_files: list[Path],
        *,
        source_label: str,
        empty_status: str,
        select_path: Path | None = None,
    ) -> None:
        """Обновляет общий плейлист для каталогов и набора отдельных файлов."""

        assert self._video_listbox is not None

        self._directory_var.set(source_label)
        self._video_files = video_files
        self._video_listbox.delete(0, tk.END)
        for path in video_files:
            self._video_listbox.insert(tk.END, path.name)

        if not video_files:
            self._selected_video_path = None
            self._preview_video_path = None
            self._preview_frame = None
            self._playlist_status_var.set(empty_status)
            self._refresh_view()
            return

        self._playlist_status_var.set(f"Файлов в плейлисте: {len(video_files)}")
        target_path = select_path.expanduser() if select_path is not None else video_files[0]
        self._select_video_path(target_path)
        if self.session is not None:
            current_video_path = Path(self.session.options.video_path).expanduser()
            if self._selected_video_path is not None and self._selected_video_path != current_video_path:
                self._close_session_only()
                self.root.title("Тепловизионный трекер")
        if self.session is None:
            self._load_preview_for_selected_video()
        self._refresh_view()

    def _load_video_directory(self, directory: Path, *, select_path: Path | None = None) -> None:
        """Сканирует папку и заполняет список роликов."""

        normalized_directory = directory.expanduser()
        if not normalized_directory.exists() or not normalized_directory.is_dir():
            messagebox.showerror("Нет папки", f"Такого каталога нет:\n{normalized_directory}", parent=self.root)
            return

        video_files = sorted(
            [
                path
                for path in normalized_directory.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            ],
            key=lambda path: path.name.lower(),
        )
        self._apply_video_playlist(
            video_files,
            source_label=str(normalized_directory),
            empty_status="В каталоге не найдено видеофайлов.",
            select_path=select_path,
        )

    def _load_video_files(self, files: list[Path], *, select_path: Path | None = None) -> None:
        """Заполняет плейлист из одного или нескольких явно выбранных файлов."""

        normalized_files = [path.expanduser() for path in files if path.suffix.lower() in VIDEO_SUFFIXES and path.exists()]
        if not normalized_files:
            messagebox.showerror("Нет видео", "Не удалось собрать ни одного подходящего видеофайла.", parent=self.root)
            return

        self._apply_video_playlist(
            normalized_files,
            source_label=f"Выбрано файлов: {len(normalized_files)}",
            empty_status="В выбранном наборе нет подходящих видеофайлов.",
            select_path=select_path or normalized_files[0],
        )

    def _select_video_path(self, path: Path) -> None:
        """Выделяет нужный ролик в списке и запоминает его как активный выбор."""

        assert self._video_listbox is not None

        target = path.expanduser()
        selected_index = 0
        for index, candidate in enumerate(self._video_files):
            if candidate == target:
                selected_index = index
                break

        self._video_listbox.selection_clear(0, tk.END)
        self._video_listbox.selection_set(selected_index)
        self._video_listbox.activate(selected_index)
        self._video_listbox.see(selected_index)
        self._selected_video_path = self._video_files[selected_index]

    def _on_video_selected(self, _event: tk.Event) -> None:
        """Обновляет выбранный ролик после щелчка в списке."""

        assert self._video_listbox is not None

        selection = self._video_listbox.curselection()
        if not selection:
            self._selected_video_path = None
            self._preview_video_path = None
            self._preview_frame = None
        else:
            index = selection[0]
            if 0 <= index < len(self._video_files):
                self._selected_video_path = self._video_files[index]
        if self.session is not None and self._selected_video_path is not None:
            current_video_path = Path(self.session.options.video_path).expanduser()
            if self._selected_video_path != current_video_path:
                self._close_session_only()
                self.root.title("Тепловизионный трекер")

        if self.session is None:
            self._load_preview_for_selected_video()
        self._refresh_view()

    def _open_selected_video(self, _event: tk.Event | None = None) -> str:
        """Открывает выбранный ролик с текущими параметрами слева."""

        if self._selected_video_path is None:
            messagebox.showerror("Нет видео", "Сначала выберите ролик в списке слева.", parent=self.root)
            return "break"

        if not self._selected_video_path.exists():
            messagebox.showerror(
                "Файл пропал",
                f"Выбранный файл больше не найден:\n{self._selected_video_path}",
                parent=self.root,
            )
            return "break"

        try:
            delay_ms = int(self._delay_var.get().strip())
        except ValueError:
            messagebox.showerror("Плохая задержка", "Задержка должна быть целым числом миллисекунд.", parent=self.root)
            return "break"

        if delay_ms < 1:
            messagebox.showerror("Плохая задержка", "Задержка должна быть не меньше 1 мс.", parent=self.root)
            return "break"

        options = LaunchOptions(
            video_path=str(self._selected_video_path),
            preset_name=self._selected_preset_name(),
            delay_ms=delay_ms,
            neural_model_path=self._model_var.get().strip() if self._model_box is not None else "",
            neural_tracker_config_path=self._tracker_var.get().strip() if self._tracker_box is not None else "",
        )
        self._open_session(options)
        return "break"

    def _open_session(self, options: LaunchOptions) -> None:
        """Закрывает старую сессию и открывает новую."""

        self._close_session_only()
        try:
            self.session = TrackingSession(options)
        except VideoOpenError as exc:
            self.session = None
            messagebox.showerror("Не удалось открыть видео", str(exc), parent=self.root)
            self._refresh_view()
            return
        except Exception as exc:
            self.session = None
            messagebox.showerror("Не удалось запустить pipeline", str(exc), parent=self.root)
            self._refresh_view()
            return

        self._selected_video_path = Path(options.video_path)
        if self._selected_source_kind() == "video_directory" and self._selected_video_path.parent.exists():
            self._directory_var.set(str(self._selected_video_path.parent))
        if self._selected_video_path in self._video_files:
            self._select_video_path(self._selected_video_path)

        self.session.advance()
        self.root.title(f"Тепловизионный трекер — {self._selected_video_path.name}")
        self._refresh_view()
        if not self.session.finished:
            self._schedule_tick(0)

    def _stop_session(self) -> str:
        """Останавливает текущее воспроизведение, не трогая список роликов."""

        self._close_session_only()
        self.root.title("Тепловизионный трекер")
        self._refresh_view()
        return "break"

    def _close_session_only(self) -> None:
        """Закрывает только текущую сессию, оставляя интерфейс живым."""

        if self._tick_after_id is not None:
            self.root.after_cancel(self._tick_after_id)
            self._tick_after_id = None
        self._close_video_writer()
        if self.session is not None:
            self.session.close()
            self.session = None
        self._last_rendered_frame = None
        self._load_preview_for_selected_video(force=True)

    def _load_preview_for_selected_video(self, *, force: bool = False) -> None:
        """Читает первый кадр выбранного ролика и держит его как превью."""

        if self._selected_source_kind() == "shared_memory":
            self._preview_video_path = None
            self._preview_frame = None
            return

        if self._selected_video_path is None:
            self._preview_video_path = None
            self._preview_frame = None
            return

        target_path = self._selected_video_path.expanduser()
        if not target_path.exists():
            self._preview_video_path = None
            self._preview_frame = None
            return

        if not force and self._preview_video_path == target_path and self._preview_frame is not None:
            return

        capture = cv2.VideoCapture(str(target_path))
        try:
            if not capture.isOpened():
                self._preview_video_path = target_path
                self._preview_frame = None
                return

            ok, frame = capture.read()
            self._preview_video_path = target_path
            self._preview_frame = frame if ok and frame is not None else None
        finally:
            capture.release()

    def _build_recording_status(self) -> str:
        """Возвращает короткое человеческое описание состояния записи."""

        if not self._record_video_var.get():
            return "выключена"
        if self._record_output_path is None:
            return "включена, файл появится после старта"
        return str(self._record_output_path)

    def _toggle_recording(self) -> None:
        """Включает или выключает запись и сразу обновляет интерфейс."""

        if not self._record_video_var.get():
            self._close_video_writer()
        self._refresh_view()

    def _ensure_video_writer(self) -> bool:
        """Лениво создаёт writer только когда действительно есть что писать."""

        if self._video_writer is not None:
            return True
        if self.session is None or self._last_rendered_frame is None:
            return False

        output_dir = PROJECT_ROOT / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(self.session.options.video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{source_path.stem}_{timestamp}.mp4"

        frame_height, frame_width = self._last_rendered_frame.shape[:2]
        fps = max(1.0, 1000.0 / max(1, self.session.safe_delay_ms))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Не удалось открыть файл для записи:\n{output_path}")

        self._video_writer = writer
        self._record_output_path = output_path
        self._last_written_render_revision = -1
        return True

    def _write_record_frame_if_needed(self) -> None:
        """Пишет текущий отрисованный кадр в файл, если запись включена."""

        if not self._record_video_var.get():
            return
        if self.session is None or self._last_rendered_frame is None:
            return
        if self.session.render_revision == self._last_written_render_revision:
            return

        try:
            if not self._ensure_video_writer():
                return
            assert self._video_writer is not None
            self._video_writer.write(self._last_rendered_frame)
            self._last_written_render_revision = self.session.render_revision
        except Exception as exc:
            self._record_video_var.set(False)
            self._close_video_writer()
            messagebox.showerror("Ошибка записи видео", str(exc), parent=self.root)

    def _close_video_writer(self) -> None:
        """Закрывает активную запись, если она была открыта."""

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._record_output_path = None
        self._last_written_render_revision = -1

    def _toggle_pause(self) -> str:
        """Переключает паузу и сразу обновляет интерфейс."""

        if self.session is None or self.session.finished:
            return "break"
        self.session.toggle_pause()
        self._refresh_view()
        self._schedule_tick(0)
        return "break"

    def _step_once(self) -> str:
        """Двигает видео ровно на один кадр вперёд."""

        if self.session is None or self.session.finished:
            return "break"
        self.session.request_step()
        self.session.advance()
        self._refresh_view()
        self._schedule_tick(0)
        return "break"

    def _reset_tracker(self) -> str:
        """Сбрасывает только трек, не трогая само видео."""

        if self.session is None:
            return "break"
        self.session.request_reset()
        self._refresh_view()
        return "break"

    def _close(self) -> str:
        """Закрывает окно и освобождает ресурсы."""

        if self._closed:
            return "break"
        self._closed = True
        self._close_session_only()
        self.root.destroy()
        return "break"

    def _on_left_click(self, event: tk.Event) -> str:
        """Запускает выбор новой цели по клику мыши."""

        point = self._map_event_to_frame(event.x, event.y)
        if point is None:
            return "break"

        click_from_preview = self.session is None
        if self.session is None:
            if self._selected_video_path is None:
                return "break"
            self._open_selected_video()
            if self.session is None:
                return "break"
            if click_from_preview:
                point = self._map_preview_point_to_session_frame(point)

        if self.session is None:
            return "break"
        self.session.request_click(point)
        self._refresh_view()
        return "break"

    def _on_right_click(self, _event: tk.Event) -> str:
        """Сбрасывает трек по правому клику."""

        if self.session is None:
            return "break"
        self.session.request_reset()
        self._refresh_view()
        return "break"

    def _map_event_to_frame(self, x: int, y: int) -> tuple[int, int] | None:
        """Переводит координаты клика на канве в координаты исходного кадра."""

        offset_x, offset_y, display_width, display_height, frame_width, frame_height = self._display_box
        if display_width <= 0 or display_height <= 0:
            return None
        if x < offset_x or y < offset_y or x >= offset_x + display_width or y >= offset_y + display_height:
            return None

        local_x = x - offset_x
        local_y = y - offset_y
        frame_x = int(local_x * frame_width / display_width)
        frame_y = int(local_y * frame_height / display_height)
        frame_x = min(max(frame_x, 0), frame_width - 1)
        frame_y = min(max(frame_y, 0), frame_height - 1)
        return frame_x, frame_y

    def _map_preview_point_to_session_frame(self, point: tuple[int, int]) -> tuple[int, int]:
        """Пересчитывает клик с превью в координаты кадра внутри активной сессии.

        Это нужно, потому что превью может быть показано в исходном размере,
        а рабочий кадр после препроцессинга уже уменьшен под текущий пресет.
        """

        if self._preview_frame is None or self.session is None or self.session.current_frame is None:
            return point

        source_height, source_width = self._preview_frame.shape[:2]
        target_height, target_width = self.session.current_frame.bgr.shape[:2]
        if source_width <= 0 or source_height <= 0:
            return point
        if source_width == target_width and source_height == target_height:
            return point

        x = int(round(point[0] * target_width / source_width))
        y = int(round(point[1] * target_height / source_height))
        x = min(max(x, 0), target_width - 1)
        y = min(max(y, 0), target_height - 1)
        return x, y
