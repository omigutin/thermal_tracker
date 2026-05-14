from __future__ import annotations

from typing import TypeAlias

from .operations import (
    BilateralBlurFramePreprocessorConfig,
    ClaheContrastFramePreprocessorConfig,
    GaussianBlurFramePreprocessorConfig,
    GradientFramePreprocessorConfig,
    MedianBlurFramePreprocessorConfig,
    MinMaxNormalizeFramePreprocessorConfig,
    PercentileNormalizeFramePreprocessorConfig,
    ResizeFramePreprocessorConfig,
    SharpnessMetricFramePreprocessorConfig,
)


FramePreprocessorConfig: TypeAlias = (
    ResizeFramePreprocessorConfig
    | GaussianBlurFramePreprocessorConfig
    | MedianBlurFramePreprocessorConfig
    | BilateralBlurFramePreprocessorConfig
    | MinMaxNormalizeFramePreprocessorConfig
    | PercentileNormalizeFramePreprocessorConfig
    | ClaheContrastFramePreprocessorConfig
    | GradientFramePreprocessorConfig
    | SharpnessMetricFramePreprocessorConfig
)


_FramePreprocessorConfigClass: TypeAlias = (
    type[ResizeFramePreprocessorConfig]
    | type[GaussianBlurFramePreprocessorConfig]
    | type[MedianBlurFramePreprocessorConfig]
    | type[BilateralBlurFramePreprocessorConfig]
    | type[MinMaxNormalizeFramePreprocessorConfig]
    | type[PercentileNormalizeFramePreprocessorConfig]
    | type[ClaheContrastFramePreprocessorConfig]
    | type[GradientFramePreprocessorConfig]
    | type[SharpnessMetricFramePreprocessorConfig]
)


FRAME_PREPROCESSOR_CONFIG_CLASSES: dict[str, _FramePreprocessorConfigClass] = {
    str(ResizeFramePreprocessorConfig.operation_type): ResizeFramePreprocessorConfig,
    str(GaussianBlurFramePreprocessorConfig.operation_type): GaussianBlurFramePreprocessorConfig,
    str(MedianBlurFramePreprocessorConfig.operation_type): MedianBlurFramePreprocessorConfig,
    str(BilateralBlurFramePreprocessorConfig.operation_type): BilateralBlurFramePreprocessorConfig,
    str(MinMaxNormalizeFramePreprocessorConfig.operation_type): MinMaxNormalizeFramePreprocessorConfig,
    str(PercentileNormalizeFramePreprocessorConfig.operation_type): PercentileNormalizeFramePreprocessorConfig,
    str(ClaheContrastFramePreprocessorConfig.operation_type): ClaheContrastFramePreprocessorConfig,
    str(GradientFramePreprocessorConfig.operation_type): GradientFramePreprocessorConfig,
    str(SharpnessMetricFramePreprocessorConfig.operation_type): SharpnessMetricFramePreprocessorConfig,
}
