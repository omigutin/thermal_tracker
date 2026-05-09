"""
    Менеджер последовательного применения атомарных операций предобработки кадра.

    Класс отвечает за:
        1. Преобразование описаний операций (enum или строка) в готовые экземпляры.
        2. Безопасную начальную сборку ProcessedFrame из сырого numpy-кадра.
        3. Последовательный прогон ProcessedFrame через цепочку атомарных операций.

    Поддерживаемые варианты входных операций:

        1. FramePreprocessorType:
            Менеджер создаёт операцию по enum-значению.

            FramePreprocessorManager(
                methods=(
                    FramePreprocessorType.RESIZE,
                    FramePreprocessorType.GAUSSIAN_BLUR,
                    FramePreprocessorType.NORMALIZE_MINMAX,
                    FramePreprocessorType.CLAHE_CONTRAST,
                    FramePreprocessorType.GRADIENT,
                    FramePreprocessorType.SHARPNESS_METRIC,
                ),
                config=preset.preprocessing,
            )

        2. str:
            Менеджер преобразует строку в FramePreprocessorType, а затем создаёт
            соответствующую операцию. Строки распознаются и по value, и по name.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import cv2
import numpy as np

from ...config import PreprocessingConfig
from ...domain.models import ProcessedFrame
from .base_frame_preprocessor import BaseFramePreprocessor
from .frame_preprocessor_type import FramePreprocessorType
from .gaussian_blur_frame_preprocessor import GaussianBlurFramePreprocessor
from .gradient_frame_preprocessor import GradientFramePreprocessor
from .median_blur_frame_preprocessor import MedianBlurFramePreprocessor
from .normalize_minmax_frame_preprocessor import NormalizeMinMaxFramePreprocessor
from .opencv_bilateral_frame_preprocessor import BilateralFramePreprocessor
from .opencv_clahe_contrast_frame_preprocessor import ClaheContrastPreprocessor
from .opencv_percentile_normalize_frame_preprocessor import PercentileNormalizePreprocessor
from .resize_frame_preprocessor import ResizeFramePreprocessor
from .sharpness_metric_frame_preprocessor import SharpnessMetricFramePreprocessor


FramePreprocessorInput: TypeAlias = FramePreprocessorType | str


class FramePreprocessorManager:
    """Последовательно применяет несколько атомарных операций предобработки кадра."""

    def __init__(
        self,
        methods: Sequence[FramePreprocessorInput],
        config: PreprocessingConfig,
    ) -> None:
        """Подготовить список операций к запуску по конфигурации пресета."""

        self._preprocessors: tuple[BaseFramePreprocessor, ...] = tuple(
            self._build_preprocessor(method, config) for method in methods
        )

    @property
    def preprocessors(self) -> tuple[BaseFramePreprocessor, ...]:
        """Готовые экземпляры атомарных операций в порядке применения."""

        return self._preprocessors

    def process(self, raw: np.ndarray) -> ProcessedFrame:
        """Преобразовать сырой кадр в готовый ProcessedFrame через цепочку операций."""

        frame = self._initial_processed_frame(raw)
        for preprocessor in self._preprocessors:
            frame = preprocessor.process(frame)
        return frame

    @staticmethod
    def _initial_processed_frame(raw: np.ndarray) -> ProcessedFrame:
        """Сделать безопасную начальную сборку каналов ProcessedFrame из сырого кадра."""

        if raw.ndim == 2:
            gray = raw
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            bgr = raw
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        normalized = gray.copy()
        gradient = np.zeros_like(gray, dtype=np.uint8)
        return ProcessedFrame(
            bgr=bgr,
            gray=gray,
            normalized=normalized,
            gradient=gradient,
            quality=None,
        )

    @classmethod
    def _build_preprocessor(
        cls,
        method: FramePreprocessorInput,
        config: PreprocessingConfig,
    ) -> BaseFramePreprocessor:
        """Создать экземпляр атомарной операции по описанию из пресета."""

        method_type = cls._normalize_method_type(method)
        if method_type == FramePreprocessorType.RESIZE:
            return ResizeFramePreprocessor(target_width=config.resize_width)
        if method_type == FramePreprocessorType.GAUSSIAN_BLUR:
            return GaussianBlurFramePreprocessor(kernel=config.gaussian_kernel)
        if method_type == FramePreprocessorType.MEDIAN_BLUR:
            return MedianBlurFramePreprocessor(kernel=config.median_kernel)
        if method_type == FramePreprocessorType.BILATERAL:
            return BilateralFramePreprocessor(
                diameter=config.bilateral_diameter,
                sigma_color=config.bilateral_sigma_color,
                sigma_space=config.bilateral_sigma_space,
            )
        if method_type == FramePreprocessorType.NORMALIZE_MINMAX:
            return NormalizeMinMaxFramePreprocessor()
        if method_type == FramePreprocessorType.CLAHE_CONTRAST:
            return ClaheContrastPreprocessor(
                clip_limit=config.clahe_clip_limit,
                tile_grid_size=config.clahe_tile_grid_size,
            )
        if method_type == FramePreprocessorType.PERCENTILE_NORMALIZE:
            return PercentileNormalizePreprocessor(
                low_percentile=config.percentile_low,
                high_percentile=config.percentile_high,
            )
        if method_type == FramePreprocessorType.GRADIENT:
            return GradientFramePreprocessor(blur_kernel=config.gradient_blur_kernel)
        if method_type == FramePreprocessorType.SHARPNESS_METRIC:
            return SharpnessMetricFramePreprocessor()
        raise ValueError(
            f"Unsupported frame preprocessor type: {method_type!r}. "
            f"Available types: {tuple(item.value for item in FramePreprocessorType)}."
        )

    @staticmethod
    def _normalize_method_type(method: FramePreprocessorInput) -> FramePreprocessorType:
        """Преобразовать строку или enum-значение в FramePreprocessorType."""

        if isinstance(method, FramePreprocessorType):
            return method
        try:
            return FramePreprocessorType(method)
        except ValueError:
            pass

        method_by_name = FramePreprocessorType.__members__.get(method.upper())
        if method_by_name is not None:
            return method_by_name

        raise ValueError(
            f"Unsupported frame preprocessor value: {method!r}. "
            f"Available values: {tuple(item.value for item in FramePreprocessorType)}. "
            f"Available names: {tuple(item.name for item in FramePreprocessorType)}."
        )
