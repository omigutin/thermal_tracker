"""Фильтрация ложных целей."""

from .base_candidate_filter import BaseTargetFilter
from .clutter_suppression_candidate_filter import ClutterSuppressionTargetFilter
from .motion_consistency_candidate_filter import MotionConsistencyTargetFilter
from .persistence_candidate_filter import PersistenceTargetFilter
from .simple_candidate_filters import (
    AreaAspectTargetFilter,
    BorderTouchTargetFilter,
    ContrastTargetFilter,
    FilterChain,
)

__all__ = [
    "AreaAspectTargetFilter",
    "BaseTargetFilter",
    "BorderTouchTargetFilter",
    "ClutterSuppressionTargetFilter",
    "ContrastTargetFilter",
    "FilterChain",
    "MotionConsistencyTargetFilter",
    "PersistenceTargetFilter",
]
