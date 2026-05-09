"""Generic-модуль машины состояний. Не содержит бизнес-логики проекта."""

from .state_machine import StateMachine
from .state_transition_error import StateTransitionError

__all__ = (
    "StateMachine",
    "StateTransitionError",
)
