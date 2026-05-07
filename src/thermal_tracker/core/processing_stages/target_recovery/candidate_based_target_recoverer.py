"""Заготовка под повторный захват через список кандидатов."""

from __future__ import annotations

from .base_target_recoverer import BaseReacquirer


class CandidateBasedReacquirer(BaseReacquirer):
    """Будущий reacquirer, который работает поверх списка объектов-кандидатов."""

    implementation_name = "candidate_based"
    is_ready = False

    def reacquire(self, frame, last_bbox, motion):
        raise NotImplementedError("Candidate-based reacquirer пока не реализован.")
