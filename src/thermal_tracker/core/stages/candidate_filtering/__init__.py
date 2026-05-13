"""
    Стадия фильтрации кандидатов на цель.
    Пакет содержит конфигурацию, фабрику, менеджер и атомарные фильтры
    для последовательного отсеивания ложных кандидатов.
"""

from .manager import CandidateFilterManager
from .type import CandidateFilterType
from .config import CANDIDATE_FILTER_CONFIG_CLASSES, CandidateFilterConfig

__all__ = (
    "CandidateFilterManager",
    "CandidateFilterType",
    "CANDIDATE_FILTER_CONFIG_CLASSES",
    "CandidateFilterConfig",
)
