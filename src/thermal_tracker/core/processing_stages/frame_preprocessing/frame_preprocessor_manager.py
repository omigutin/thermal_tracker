"""Менеджер выбора препроцессора кадра."""

from __future__ import annotations

import numpy as np

from ...config import PreprocessingConfig
from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .frame_preprocessor_type import FramePreprocessorType
from .identity_frame_preprocessor import IdentityFramePreprocessor
from .opencv_bilateral_frame_preprocessor import BilateralFramePreprocessor
from .opencv_clahe_contrast_frame_preprocessor import ClaheContrastPreprocessor
from .opencv_percentile_normalize_frame_preprocessor import PercentileNormalizePreprocessor
from .opencv_thermal_frame_preprocessor import ThermalFramePreprocessor


FramePreprocessorInput = FramePreprocessorType | str


class FramePreprocessorManager:
    """Создаёт и запускает выбранный препроцессор кадра."""

    def __init__(self, preprocessor: FramePreprocessorInput, config: PreprocessingConfig) -> None:
        self._preprocessor: BaseFramePreprocessor = self._build_preprocessor(preprocessor, config)

    @property
    def preprocessor(self) -> BaseFramePreprocessor:
        """Возвращает подготовленный препроцессор."""

        return self._preprocessor

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Преобразует сырой кадр в ProcessedFrame."""

        return self._preprocessor.process(frame)

    @classmethod
    def _build_preprocessor(cls,
        preprocessor: FramePreprocessorInput,
        config: PreprocessingConfig,
    ) -> BaseFramePreprocessor:
        preprocessor_type = cls._normalize_preprocessor_type(preprocessor)
        if preprocessor_type == FramePreprocessorType.IDENTITY:
            return IdentityFramePreprocessor(config.resize_width)
        if preprocessor_type == FramePreprocessorType.THERMAL:
            return ThermalFramePreprocessor(config)
        if preprocessor_type == FramePreprocessorType.BILATERAL:
            return BilateralFramePreprocessor(config.resize_width)
        if preprocessor_type == FramePreprocessorType.CLAHE_CONTRAST:
            return ClaheContrastPreprocessor(
                resize_width=config.resize_width,
                clip_limit=config.clahe_clip_limit,
                tile_grid_size=config.clahe_tile_grid_size,
            )
        if preprocessor_type == FramePreprocessorType.PERCENTILE_NORMALIZE:
            return PercentileNormalizePreprocessor(config.resize_width)
        raise ValueError(f"Unsupported frame preprocessor type: {preprocessor_type!r}.")

    @staticmethod
    def _normalize_preprocessor_type(preprocessor: FramePreprocessorInput) -> FramePreprocessorType:
        if isinstance(preprocessor, FramePreprocessorType):
            return preprocessor
        try:
            return FramePreprocessorType(preprocessor)
        except ValueError:
            pass
        preprocessor_by_name = FramePreprocessorType.__members__.get(preprocessor.upper())
        if preprocessor_by_name is not None:
            return preprocessor_by_name
        raise ValueError(
            f"Unsupported frame preprocessor value: {preprocessor!r}. "
            f"Available values: {tuple(item.value for item in FramePreprocessorType)}."
        )
