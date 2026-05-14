"""Базовый контракт стадии повторного захвата потерянной цели.

Recoverer хранит знание о том, как выглядит цель, и пытается вернуть её
на новом кадре. Минимальный recoverer обязан реализовать только
:meth:`reacquire`. Методы :meth:`remember` и :meth:`reset` имеют
дефолтную пустую реализацию: конкретная стратегия может их переопределить
для ведения внутреннего состояния (шаблонов, гистограмм, истории).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from thermal_tracker.core.domain.models import BoundingBox, ProcessedFrame
from thermal_tracker.core.stages.frame_stabilization.result import FrameStabilizerResult


class BaseReacquirer(ABC):
    """Базовый класс для всех recoverer-ов стадии target_recovery."""

    @abstractmethod
    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: FrameStabilizerResult,
        lost_frames: int = 0,
    ) -> BoundingBox | None:
        """Вернуть bbox цели на новом кадре или ``None``, если найти не удалось.

        ``lost_frames`` — сколько кадров подряд цель уже потеряна. Конкретные
        реализации могут использовать это, чтобы расширять зону поиска
        по мере затягивания потери.
        """

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Запомнить, как сейчас выглядит уверенно сопровождаемая цель.

        Pipeline вызывает этот метод на каждом подтверждённом TRACKING-кадре.
        Конкретные реализации могут вести adaptive template, гистограмму,
        embedding и т.п. По умолчанию ничего не делает.
        """

    def reset(self) -> None:
        """Сбросить внутреннее состояние recoverer-а.

        Pipeline вызывает этот метод при reset-команде или новом клике.
        По умолчанию ничего не делает.
        """
