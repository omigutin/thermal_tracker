"""Небольшие вспомогательные подсказки для Tkinter."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable


class HoverTooltip:
    """Показывает маленькую подсказку рядом с виджетом при наведении."""

    def __init__(self, widget: tk.Widget, text_provider: Callable[[], str], delay_ms: int = 350) -> None:
        self.widget = widget
        self.text_provider = text_provider
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")

    def _on_enter(self, event: tk.Event) -> None:
        self._schedule(event)

    def _on_leave(self, _event: tk.Event) -> None:
        self._cancel()
        self._hide()

    def _on_motion(self, event: tk.Event) -> None:
        if self._window is not None:
            self._place_window(event.x_root, event.y_root)

    def _schedule(self, event: tk.Event) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, lambda: self._show(event))

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self, event: tk.Event) -> None:
        text = self.text_provider().strip()
        if not text:
            return

        if self._window is None:
            self._window = tk.Toplevel(self.widget)
            self._window.wm_overrideredirect(True)
            self._window.attributes("-topmost", True)
            label = tk.Label(
                self._window,
                text=text,
                justify="left",
                background="#fff7d6",
                foreground="#222222",
                relief="solid",
                borderwidth=1,
                padx=8,
                pady=6,
                wraplength=320,
            )
            label.pack()
        else:
            label = self._window.winfo_children()[0]
            if isinstance(label, tk.Label):
                label.configure(text=text)

        self._place_window(event.x_root, event.y_root)

    def _place_window(self, x_root: int, y_root: int) -> None:
        if self._window is None:
            return
        self._window.geometry(f"+{x_root + 16}+{y_root + 12}")

    def _hide(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None
