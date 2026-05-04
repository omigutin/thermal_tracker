"""Новый контур трекинга тепловизионных объектов по клику."""

from __future__ import annotations


def main() -> None:
    """Точка входа приложения.

    Импорт отложен, чтобы `import thermal_tracker` не тянул GUI/OpenCV.
    """

    from .gui.app import run_gui as _main

    _main()

__all__ = ["main"]
