"""Исключение, которое бросает StateMachine при попытке нелегального перехода."""

from __future__ import annotations


class StateTransitionError(Exception):
    """Бросается, когда переход между состояниями не разрешён таблицей переходов."""
