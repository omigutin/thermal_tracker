from __future__ import annotations

from typing import TypeAlias

from .operations import (
    FeatureAffineFrameStabilizerConfig,
    PhaseCorrelationFrameStabilizerConfig,
)


FrameStabilizerConfig: TypeAlias = (
    PhaseCorrelationFrameStabilizerConfig
    | FeatureAffineFrameStabilizerConfig
)


_FrameStabilizerConfigClass: TypeAlias = (
    type[PhaseCorrelationFrameStabilizerConfig]
    | type[FeatureAffineFrameStabilizerConfig]
)


FRAME_STABILIZER_CONFIG_CLASSES: dict[str, _FrameStabilizerConfigClass] = {
    str(PhaseCorrelationFrameStabilizerConfig.operation_type): PhaseCorrelationFrameStabilizerConfig,
    str(FeatureAffineFrameStabilizerConfig.operation_type): FeatureAffineFrameStabilizerConfig,
}
