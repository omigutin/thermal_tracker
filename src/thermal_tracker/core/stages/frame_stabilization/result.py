from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameStabilizerResult:
    """Грубая оценка движения камеры между соседними кадрами."""

    dx: float = 0.0
    dy: float = 0.0
    response: float = 0.0
    valid: bool = False
