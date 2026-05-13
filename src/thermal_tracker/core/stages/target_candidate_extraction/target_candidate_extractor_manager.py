"""Менеджер выбора метода сборки кандидатов на цель."""

from __future__ import annotations

from ...domain.models import DetectedObject, MotionDetectionResult, ProcessedFrame
from .base_candidate_extractor import BaseObjectBuilder
from .opencv_connected_components_candidate_extractor import ConnectedComponentsObjectBuilder
from .opencv_contour_candidate_extractor import ContourObjectBuilder
from .target_candidate_extractor_type import TargetCandidateExtractorType


TargetCandidateExtractorInput = TargetCandidateExtractorType | str


class TargetCandidateExtractorManager:
    """Создаёт и запускает выбранный сборщик кандидатов."""

    def __init__(self, extractor: TargetCandidateExtractorInput) -> None:
        self._extractor_type = self._normalize_extractor_type(extractor)
        self._extractor = self._build_extractor(self._extractor_type)

    @property
    def extractor(self) -> BaseObjectBuilder:
        """Возвращает подготовленный сборщик кандидатов."""

        return self._extractor

    def build(self, frame: ProcessedFrame, detection: MotionDetectionResult) -> list[DetectedObject]:
        """Строит кандидатов на цель из результата детектора."""

        objects = self._extractor.build(frame, detection)
        for detected_object in objects:
            detected_object.source = self._extractor_type.value
        return objects

    @classmethod
    def _build_extractor(cls, extractor_type: TargetCandidateExtractorType) -> BaseObjectBuilder:
        if extractor_type == TargetCandidateExtractorType.OPENCV_CONNECTED_COMPONENTS:
            return ConnectedComponentsObjectBuilder()
        if extractor_type == TargetCandidateExtractorType.OPENCV_CONTOUR:
            return ContourObjectBuilder()
        raise ValueError(f"Unsupported target candidate extractor type: {extractor_type!r}.")

    @staticmethod
    def _normalize_extractor_type(extractor: TargetCandidateExtractorInput) -> TargetCandidateExtractorType:
        if isinstance(extractor, TargetCandidateExtractorType):
            return extractor
        try:
            return TargetCandidateExtractorType(extractor)
        except ValueError:
            pass
        extractor_by_name = TargetCandidateExtractorType.__members__.get(extractor.upper())
        if extractor_by_name is not None:
            return extractor_by_name
        raise ValueError(
            f"Unsupported target candidate extractor value: {extractor!r}. "
            f"Available values: {tuple(item.value for item in TargetCandidateExtractorType)}."
        )
