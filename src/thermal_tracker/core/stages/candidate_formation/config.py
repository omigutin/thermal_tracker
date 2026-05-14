from __future__ import annotations

from typing import TypeAlias

from .operations import (
    ConnectedComponentsCandidateFormerConfig,
    ContourCandidateFormerConfig,
)

CandidateFormerConfig: TypeAlias = (
    ConnectedComponentsCandidateFormerConfig
    | ContourCandidateFormerConfig
)


_CandidateFormerConfigClass: TypeAlias = (
    type[ConnectedComponentsCandidateFormerConfig]
    | type[ContourCandidateFormerConfig]
)


CANDIDATE_FORMER_CONFIG_CLASSES: dict[str, _CandidateFormerConfigClass] = {
    str(ConnectedComponentsCandidateFormerConfig.operation_type): (
        ConnectedComponentsCandidateFormerConfig
    ),
    str(ContourCandidateFormerConfig.operation_type): ContourCandidateFormerConfig,
}
