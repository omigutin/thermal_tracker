from __future__ import annotations

from typing import TypeAlias

from .operations.area_aspect_candidate_filter import AreaAspectCandidateFilterConfig
from .operations.border_touch_candidate_filter import BorderTouchCandidateFilterConfig
from .operations.contrast_candidate_filter import ContrastCandidateFilterConfig


"""Допустимые конфигурации атомарных фильтров кандидатов."""
CandidateFilterConfig: TypeAlias = (
    AreaAspectCandidateFilterConfig
    | BorderTouchCandidateFilterConfig
    | ContrastCandidateFilterConfig
)


"""Допустимые классы конфигураций атомарных фильтров кандидатов."""
_CandidateFilterConfigClass: TypeAlias = (
    type[AreaAspectCandidateFilterConfig]
    | type[BorderTouchCandidateFilterConfig]
    | type[ContrastCandidateFilterConfig]
)


"""Связь TOML-значения `type` с классом конфигурации фильтра."""
CANDIDATE_FILTER_CONFIG_CLASSES: dict[str, _CandidateFilterConfigClass] = {
    str(AreaAspectCandidateFilterConfig.filter_type): AreaAspectCandidateFilterConfig,
    str(BorderTouchCandidateFilterConfig.filter_type): BorderTouchCandidateFilterConfig,
    str(ContrastCandidateFilterConfig.filter_type): ContrastCandidateFilterConfig,
}
