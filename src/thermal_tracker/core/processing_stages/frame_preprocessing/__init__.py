"""Предобработка изображения."""

from .agc_compensation_frame_preprocessor import AgcCompensationPreprocessor
from .base_frame_preprocessor import BaseFramePreprocessor
from .opencv_bilateral_frame_preprocessor import BilateralFramePreprocessor
from .opencv_clahe_contrast_frame_preprocessor import ClaheContrastPreprocessor
from .gradient_enhanced_frame_preprocessor import GradientEnhancedPreprocessor
from .identity_frame_preprocessor import IdentityFramePreprocessor
from .opencv_percentile_normalize_frame_preprocessor import PercentileNormalizePreprocessor
from .temporal_denoise_frame_preprocessor import TemporalDenoisePreprocessor
from .opencv_thermal_frame_preprocessor import ThermalFramePreprocessor

__all__ = [
    "AgcCompensationPreprocessor",
    "BaseFramePreprocessor",
    "BilateralFramePreprocessor",
    "ClaheContrastPreprocessor",
    "GradientEnhancedPreprocessor",
    "IdentityFramePreprocessor",
    "PercentileNormalizePreprocessor",
    "TemporalDenoisePreprocessor",
    "ThermalFramePreprocessor",
]
