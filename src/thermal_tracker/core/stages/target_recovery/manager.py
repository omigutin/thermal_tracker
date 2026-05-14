"""Менеджер выбора и запуска стадии повторного захвата цели.

Принимает тип recoverer-а и общую конфигурацию ``TargetRecoveryConfig``,
из которой берёт нужные параметры для конкретной реализации. Проксирует
методы базового контракта (``remember``, ``reacquire``, ``reset``) на
выбранный recoverer.
"""

from __future__ import annotations

from typing import TypeAlias

from ...config.preset import TargetRecoveryConfig
from ...domain.models import BoundingBox, ProcessedFrame
from ..frame_stabilization.result import FrameStabilizerResult
from thermal_tracker.core.stages.target_recovery.operations.base_target_recoverer import BaseReacquirer
from .candidate_based_target_recoverer import CandidateBasedReacquirer
from .global_search_target_recoverer import GlobalReacquirer
from thermal_tracker.core.stages.target_recovery.operations.irst_contrast_target_recoverer import IrstContrastRecoverer
from thermal_tracker.core.stages.target_recovery.operations.local_template_target_recoverer import LocalTemplateReacquirer
from .multi_scale_target_recoverer import MultiScaleReacquirer
from .type import TargetRecovererType


TargetRecovererInput: TypeAlias = TargetRecovererType | str


class TargetRecovererManager:
    """Создаёт и запускает выбранный метод повторного захвата."""

    def __init__(
        self,
        recoverer: TargetRecovererInput,
        config: TargetRecoveryConfig | None = None,
    ) -> None:
        self._config = config or TargetRecoveryConfig()
        self._recoverer = self._build_recoverer(recoverer, self._config)

    @property
    def recoverer(self) -> BaseReacquirer:
        """Возвращает подготовленный recoverer."""

        return self._recoverer

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Передаёт recoverer-у уверенно сопровождаемую цель для обновления памяти."""

        self._recoverer.remember(frame, bbox)

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: FrameStabilizerResult,
        lost_frames: int = 0,
    ) -> BoundingBox | None:
        """Пытается вернуть потерянную цель."""

        return self._recoverer.reacquire(frame, last_bbox, motion, lost_frames)

    def reset(self) -> None:
        """Сбрасывает внутреннее состояние выбранного recoverer-а."""

        self._recoverer.reset()

    @classmethod
    def _build_recoverer(
        cls,
        recoverer: TargetRecovererInput,
        config: TargetRecoveryConfig,
    ) -> BaseReacquirer:
        recoverer_type = cls._normalize_recoverer_type(recoverer)
        if recoverer_type == TargetRecovererType.LOCAL_TEMPLATE:
            return LocalTemplateReacquirer(
                search_padding=config.search_padding,
                search_padding_growth=config.search_padding_growth,
                scales=config.scales,
                match_threshold=config.match_threshold,
                template_alpha=config.template_alpha,
            )
        if recoverer_type == TargetRecovererType.GLOBAL_SEARCH:
            return GlobalReacquirer()
        if recoverer_type == TargetRecovererType.CANDIDATE_BASED:
            return CandidateBasedReacquirer()
        if recoverer_type == TargetRecovererType.MULTI_SCALE:
            return MultiScaleReacquirer()
        if recoverer_type == TargetRecovererType.IRST_CONTRAST:
            return IrstContrastRecoverer(
                search_padding=config.search_padding,
                search_padding_growth=config.search_padding_growth,
                max_speed_px_per_frame=config.max_speed_px_per_frame,
            )
        raise ValueError(f"Unsupported target recoverer type: {recoverer_type!r}.")

    @staticmethod
    def _normalize_recoverer_type(recoverer: TargetRecovererInput) -> TargetRecovererType:
        if isinstance(recoverer, TargetRecovererType):
            return recoverer
        try:
            return TargetRecovererType(recoverer)
        except ValueError:
            pass
        recoverer_by_name = TargetRecovererType.__members__.get(recoverer.upper())
        if recoverer_by_name is not None:
            return recoverer_by_name
        raise ValueError(
            f"Unsupported target recoverer value: {recoverer!r}. "
            f"Available values: {tuple(item.value for item in TargetRecovererType)}."
        )
