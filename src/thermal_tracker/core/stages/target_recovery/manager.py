from __future__ import annotations

from collections.abc import Sequence

from ...domain.models import BoundingBox, ProcessedFrame
from ..frame_stabilization import FrameStabilizerResult
from .config import TargetRecovererConfig
from .factory import TargetRecovererFactory
from .operations import BaseTargetRecoverer
from .result import TargetRecoveryResult


class TargetRecoveryManager:
    """Управляет выполнением операций повторного захвата цели."""

    def __init__(self, operations: Sequence[TargetRecovererConfig]) -> None:
        """Создать менеджер и подготовить активные recoverer-ы."""
        self._recoverers: tuple[BaseTargetRecoverer, ...] = (
            TargetRecovererFactory.build_many(operations)
        )

    @property
    def recoverers(self) -> tuple[BaseTargetRecoverer, ...]:
        """Вернуть подготовленные runtime recoverer-ы."""
        return self._recoverers

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Передать уверенно найденную цель всем recoverer-ам."""
        for recoverer in self._recoverers:
            recoverer.remember(frame=frame, bbox=bbox)

    def recover(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: FrameStabilizerResult,
        lost_frames: int = 0,
    ) -> TargetRecoveryResult:
        """Попытаться восстановить потерянную цель."""
        if not self._recoverers:
            return TargetRecoveryResult(
                bbox=None,
                source_name="",
                message="Target recovery is disabled.",
            )

        last_result: TargetRecoveryResult | None = None

        for recoverer in self._recoverers:
            result = recoverer.recover(
                frame=frame,
                last_bbox=last_bbox,
                motion=motion,
                lost_frames=lost_frames,
            )
            last_result = result

            if result.recovered:
                return result

        if last_result is not None:
            return last_result

        return TargetRecoveryResult(
            bbox=None,
            source_name="",
            message="Target was not recovered.",
        )

    def reset(self) -> None:
        """Сбросить внутреннее состояние всех recoverer-ов."""
        for recoverer in self._recoverers:
            recoverer.reset()
