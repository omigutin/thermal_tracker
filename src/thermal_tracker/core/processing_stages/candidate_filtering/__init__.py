"""
    Публичный API стадии фильтрации кандидатов.
    Наружу экспортируются только:
        - CandidateFilterType: публичные идентификаторы доступных фильтров;
        - CandidateFilterManager: менеджер создания и последовательного запуска фильтров.
    Конкретные реализации фильтров считаются внутренними деталями модуля.
    Внешний код и пресеты не должны импортировать их напрямую.
"""

from .candidate_filter_manager import CandidateFilterManager
from .candidate_filter_type import CandidateFilterType

__all__ = (
    "CandidateFilterManager",
    "CandidateFilterType",
)
