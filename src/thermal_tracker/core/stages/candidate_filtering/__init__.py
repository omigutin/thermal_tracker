"""
    Стадия фильтрации кандидатов на цель.
    Пакет содержит конфигурацию, фабрику, менеджер и атомарные фильтры
    для последовательного отсеивания ложных кандидатов.
"""

from .manager import CandidateFilterManager
from .type import CandidateFilterType

__all__ = (
    "CandidateFilterManager",
    "CandidateFilterType",
)
