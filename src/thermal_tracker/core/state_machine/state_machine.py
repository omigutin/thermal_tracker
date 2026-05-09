"""
Простая generic-машина состояний без бизнес-логики.

Модуль хранит текущее состояние и легализует переходы по необязательной
таблице переходов. Никаких знаний о доменных моделях проекта не имеет.

Когда понадобится более развитая state machine, этот модуль можно заменить
на внешнюю библиотеку без изменений зависимых модулей, при условии, что
внешняя библиотека предоставит совместимый минимальный контракт:
``current``, ``can_transition_to``, ``transition_to``, ``reset``.
"""

from __future__ import annotations

from collections.abc import Mapping, Set as AbstractSet
from typing import Generic, TypeVar

from .state_transition_error import StateTransitionError

StateT = TypeVar("StateT")


class StateMachine(Generic[StateT]):
    """Хранит текущее состояние и легализует переходы между состояниями."""

    def __init__(
        self,
        initial: StateT,
        transitions: Mapping[StateT, AbstractSet[StateT]] | None = None,
    ) -> None:
        """
        :param initial: Начальное состояние машины.
        :param transitions: Таблица разрешённых переходов вида
            ``{состояние: {разрешённые следующие состояния}}``.
            Если ``None``, разрешены любые переходы.
        """

        self._current: StateT = initial
        self._initial: StateT = initial
        self._transitions: Mapping[StateT, AbstractSet[StateT]] | None = transitions

    @property
    def current(self) -> StateT:
        """Текущее состояние машины."""

        return self._current

    def can_transition_to(self, new_state: StateT) -> bool:
        """Возвращает ``True``, если переход из текущего состояния в ``new_state`` разрешён."""

        if self._transitions is None:
            return True
        allowed = self._transitions.get(self._current)
        if allowed is None:
            return False
        return new_state in allowed

    def transition_to(self, new_state: StateT) -> None:
        """Переводит машину в ``new_state``.

        Бросает :class:`StateTransitionError`, если переход не разрешён таблицей.
        """

        if not self.can_transition_to(new_state):
            raise StateTransitionError(
                f"Transition from {self._current!r} to {new_state!r} is not allowed."
            )
        self._current = new_state

    def reset(self, to: StateT | None = None) -> None:
        """Сбрасывает машину в начальное или явно указанное состояние.

        Сброс не проверяет таблицу переходов и предназначен для жёсткого
        возврата машины в безопасное состояние, например при ошибке или
        внешнем reset-команде.
        """

        self._current = self._initial if to is None else to
