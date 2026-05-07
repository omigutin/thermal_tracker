"""Публичный API стадии сборки кандидатов на цель."""

from .target_candidate_extractor_manager import TargetCandidateExtractorManager
from .target_candidate_extractor_type import TargetCandidateExtractorType

__all__ = (
    "TargetCandidateExtractorManager",
    "TargetCandidateExtractorType",
)
