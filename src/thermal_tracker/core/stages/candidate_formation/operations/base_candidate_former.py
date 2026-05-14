from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.models import ProcessedFrame
from ..result import CandidateFormerResult
from ...motion_localization import MotionLocalizerResult


class BaseCandidateFormer(ABC):
    """Базовый интерфейс операции формирования кандидатов."""

    @abstractmethod
    def apply(
        self,
        frame: ProcessedFrame,
        motion_localizer_result: MotionLocalizerResult,
    ) -> tuple[CandidateFormerResult, ...]:
        """Сформировать кандидатов по результату локализации движения."""
        raise NotImplementedError
