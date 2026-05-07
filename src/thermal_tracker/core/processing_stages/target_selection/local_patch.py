"""Локальный фрагмент кадра для выбора цели по клику."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalPatch:
    """Локальный кусок кадра вокруг клика или текущего бокса."""

    image: np.ndarray
    origin_x: int
    origin_y: int
    local_x: int
    local_y: int
