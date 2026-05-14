from __future__ import annotations

from typing import TypeAlias

from .operations.frame_difference_motion_localizer import FrameDifferenceMotionLocalizerConfig
from .operations.knn_motion_localizer import KnnMotionLocalizerConfig
from .operations.mog2_motion_localizer import Mog2MotionLocalizerConfig
from .operations.running_average_motion_localizer import RunningAverageMotionLocalizerConfig


MotionLocalizationConfig: TypeAlias = (
        FrameDifferenceMotionLocalizerConfig
        | KnnMotionLocalizerConfig
        | Mog2MotionLocalizerConfig
        | RunningAverageMotionLocalizerConfig
)


_MotionLocalizerConfigClass: TypeAlias = (
    type[FrameDifferenceMotionLocalizerConfig]
    | type[KnnMotionLocalizerConfig]
    | type[Mog2MotionLocalizerConfig]
    | type[RunningAverageMotionLocalizerConfig]
)


MOTION_LOCALIZER_CONFIG_CLASSES: dict[str, _MotionLocalizerConfigClass] = {
    str(FrameDifferenceMotionLocalizerConfig.operation_type): FrameDifferenceMotionLocalizerConfig,
    str(KnnMotionLocalizerConfig.operation_type): KnnMotionLocalizerConfig,
    str(Mog2MotionLocalizerConfig.operation_type): Mog2MotionLocalizerConfig,
    str(RunningAverageMotionLocalizerConfig.operation_type): RunningAverageMotionLocalizerConfig,
}
