from __future__ import annotations

from dataclasses import dataclass, field

from .....domain.models import BoundingBox
from .template_point_config import TemplatePointTargetTrackerConfig


@dataclass(slots=True)
class EdgeExitGuard:
    """Отслеживает возможный выход цели за край кадра."""

    config: TemplatePointTargetTrackerConfig
    exit_edges: set[str] = field(default_factory=set, init=False)

    @property
    def has_exit_edges(self) -> bool:
        """Проверить, есть ли признаки выхода цели за край кадра."""
        return bool(self.exit_edges)

    def reset(self) -> None:
        """Сбросить сохранённые стороны выхода цели."""
        self.exit_edges.clear()

    def update_exit_edges(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        """Запомнить стороны кадра, к которым цель подошла вплотную."""
        self.exit_edges = self.detect_edge_contact(bbox=bbox, frame_shape=frame_shape)

    def is_invalid_edge_candidate(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> bool:
        """Проверить, не перескочил ли трек после выхода за край на похожую цель."""
        if not self.exit_edges:
            return False

        candidate_edges = self.detect_edge_contact(bbox=bbox, frame_shape=frame_shape)
        return not self.exit_edges.intersection(candidate_edges)

    def detect_edge_contact(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> set[str]:
        """Вернуть стороны кадра, которых касается bbox."""
        frame_height, frame_width = frame_shape[:2]
        margin = max(0, int(self.config.edge_exit_margin))
        edges: set[str] = set()

        if bbox.x <= margin:
            edges.add("left")
        if bbox.y <= margin:
            edges.add("top")
        if bbox.x2 >= frame_width - margin:
            edges.add("right")
        if bbox.y2 >= frame_height - margin:
            edges.add("bottom")

        return edges
    