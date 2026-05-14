from __future__ import annotations

from dataclasses import dataclass

from .....domain.models import BoundingBox
from .template_point_config import TemplatePointTargetTrackerConfig


@dataclass(slots=True)
class BboxStabilizer:
    """Стабилизирует размер bbox между соседними кадрами."""

    config: TemplatePointTargetTrackerConfig

    def stabilize(
        self,
        measured_bbox: BoundingBox,
        previous_bbox: BoundingBox | None,
        canonical_size: tuple[int, int] | None,
        lost_frames: int,
        frame_shape: tuple[int, int] | tuple[int, int, int],
    ) -> BoundingBox:
        """Ограничить резкие скачки размера bbox."""
        if previous_bbox is None:
            return measured_bbox

        if lost_frames == 0:
            min_scale = self.config.max_size_shrink
            max_scale = self.config.max_size_growth
        else:
            min_scale = self.config.max_size_shrink_on_reacquire
            max_scale = self.config.max_size_growth_on_reacquire

        min_width = max(self.config.min_box_size, int(round(previous_bbox.width * min_scale)))
        max_width = max(min_width, int(round(previous_bbox.width * max_scale)))
        min_height = max(self.config.min_box_size, int(round(previous_bbox.height * min_scale)))
        max_height = max(min_height, int(round(previous_bbox.height * max_scale)))

        stabilized_width = min(max(measured_bbox.width, min_width), max_width)
        stabilized_height = min(max(measured_bbox.height, min_height), max_height)

        if canonical_size is not None:
            initial_width, initial_height = canonical_size
            initial_growth = self.allowed_growth_from_initial(
                initial_width=initial_width,
                initial_height=initial_height,
            )
            max_initial_width = max(
                self.config.min_box_size,
                int(round(initial_width * initial_growth)),
            )
            max_initial_height = max(
                self.config.min_box_size,
                int(round(initial_height * initial_growth)),
            )
            stabilized_width = min(stabilized_width, max_initial_width)
            stabilized_height = min(stabilized_height, max_initial_height)

        center_x, center_y = measured_bbox.center

        return BoundingBox.from_center(
            center_x,
            center_y,
            stabilized_width,
            stabilized_height,
        ).clamp(frame_shape)

    def allowed_growth_from_initial(self, initial_width: int, initial_height: int) -> float:
        """Вернуть допустимый рост bbox относительно стартового размера."""
        configured_growth = max(1.0, float(self.config.max_size_growth_from_initial))
        initial_max_side = max(initial_width, initial_height)

        if initial_max_side < 24:
            return min(configured_growth, 2.0)
        if initial_max_side < 40:
            return min(configured_growth, 1.45)
        if initial_max_side < 70:
            return min(configured_growth, 1.2)

        return configured_growth
    