"""Менеджер выбора метода повторного захвата цели."""

from __future__ import annotations

from ...domain.models import BoundingBox, GlobalMotion, ProcessedFrame
from .base_target_recoverer import BaseReacquirer
from .candidate_based_target_recoverer import CandidateBasedReacquirer
from .global_search_target_recoverer import GlobalReacquirer
from .local_template_target_recoverer import LocalTemplateReacquirer
from .multi_scale_target_recoverer import MultiScaleReacquirer
from .target_recoverer_type import TargetRecovererType


TargetRecovererInput = TargetRecovererType | str


class TargetRecovererManager:
    """Создаёт и запускает выбранный метод повторного захвата."""

    def __init__(self, recoverer: TargetRecovererInput) -> None:
        self._recoverer = self._build_recoverer(recoverer)

    @property
    def recoverer(self) -> BaseReacquirer:
        """Возвращает подготовленный recoverer."""

        return self._recoverer

    def reacquire(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: GlobalMotion,
    ) -> BoundingBox | None:
        """Пытается вернуть потерянную цель."""

        return self._recoverer.reacquire(frame, last_bbox, motion)

    @classmethod
    def _build_recoverer(cls, recoverer: TargetRecovererInput) -> BaseReacquirer:
        recoverer_type = cls._normalize_recoverer_type(recoverer)
        if recoverer_type == TargetRecovererType.LOCAL_TEMPLATE:
            return LocalTemplateReacquirer()
        if recoverer_type == TargetRecovererType.GLOBAL_SEARCH:
            return GlobalReacquirer()
        if recoverer_type == TargetRecovererType.CANDIDATE_BASED:
            return CandidateBasedReacquirer()
        if recoverer_type == TargetRecovererType.MULTI_SCALE:
            return MultiScaleReacquirer()
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
