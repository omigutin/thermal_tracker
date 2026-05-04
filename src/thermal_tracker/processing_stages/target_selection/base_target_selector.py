"""Базовый класс для инициализации цели по клику."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...domain.models import BoundingBox, ProcessedFrame, SelectionResult


class BaseClickInitializer(ABC):
    """Любая реализация должна уметь найти объект вокруг точки клика."""

    implementation_name = "base"
    is_ready = False

    @abstractmethod
    def select(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> SelectionResult:
        """Находит цель вокруг точки клика или уточняет уже известную цель."""
