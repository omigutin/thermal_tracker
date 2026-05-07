"""Менеджер выбора метода стабилизации кадра."""

from __future__ import annotations

from ...config import GlobalMotionConfig
from ...domain.models import GlobalMotion, ProcessedFrame
from .base_stabilizer import BaseMotionEstimator
from .frame_stabilizer_type import FrameStabilizerType
from .no_stabilizer import NoMotionEstimator
from .opencv_feature_affine_stabilizer import FeatureAffineMotionEstimator
from .opencv_phase_correlation_stabilizer import PhaseCorrelationMotionEstimator


FrameStabilizerInput = FrameStabilizerType | str


class FrameStabilizerManager:
    """Создаёт и запускает выбранный оценщик движения камеры."""

    def __init__(self, stabilizer: FrameStabilizerInput, config: GlobalMotionConfig) -> None:
        self._estimator = self._build_estimator(stabilizer, config)

    @property
    def estimator(self) -> BaseMotionEstimator:
        """Возвращает подготовленный оценщик движения."""

        return self._estimator

    def estimate(self, frame: ProcessedFrame) -> GlobalMotion:
        """Оценивает глобальное движение кадра."""

        return self._estimator.estimate(frame)

    @classmethod
    def _build_estimator(
        cls,
        stabilizer: FrameStabilizerInput,
        config: GlobalMotionConfig,
    ) -> BaseMotionEstimator:
        if not config.enabled:
            return NoMotionEstimator()
        stabilizer_type = cls._normalize_stabilizer_type(stabilizer)
        if stabilizer_type == FrameStabilizerType.NO:
            return NoMotionEstimator()
        if stabilizer_type == FrameStabilizerType.OPENCV_PHASE_CORRELATION:
            return PhaseCorrelationMotionEstimator(config)
        if stabilizer_type == FrameStabilizerType.OPENCV_FEATURE_AFFINE:
            return FeatureAffineMotionEstimator(max_shift_ratio=config.max_shift_ratio)
        raise ValueError(f"Unsupported frame stabilizer type: {stabilizer_type!r}.")

    @staticmethod
    def _normalize_stabilizer_type(stabilizer: FrameStabilizerInput) -> FrameStabilizerType:
        if isinstance(stabilizer, FrameStabilizerType):
            return stabilizer
        try:
            return FrameStabilizerType(stabilizer)
        except ValueError:
            pass
        stabilizer_by_name = FrameStabilizerType.__members__.get(stabilizer.upper())
        if stabilizer_by_name is not None:
            return stabilizer_by_name
        raise ValueError(
            f"Unsupported frame stabilizer value: {stabilizer!r}. "
            f"Available values: {tuple(item.value for item in FrameStabilizerType)}."
        )
