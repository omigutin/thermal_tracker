"""Базовый класс для источников кадров.

Здесь идея простая: у любой реализации должно быть одинаковое лицо.
Не важно, читаем мы видеофайл, папку с картинками или поток с камеры,
остальной код хочет знать только две вещи:
1. как получить следующий кадр;
2. как аккуратно освободить ресурс.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseFrameSource(ABC):
    """Общий контракт для всех источников кадров."""

    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Читает следующий кадр.

        Возвращает `False, None`, когда кадры закончились или источник временно не дал данных.
        """

    @abstractmethod
    def close(self) -> None:
        """Освобождает все внутренние ресурсы источника."""
