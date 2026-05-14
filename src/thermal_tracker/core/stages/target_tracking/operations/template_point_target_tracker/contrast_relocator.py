from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....domain.models import BoundingBox, ProcessedFrame, TrackerState
from ....target_selection import TargetPolarity
from ....target_selection.manager import TargetSelectionManager
from .search_result import SearchResult
from .template_point_config import TemplatePointTargetTrackerConfig


@dataclass(slots=True)
class ContrastRelocator:
    """Пробует вернуть цель через contrast selector рядом с прогнозом."""

    config: TemplatePointTargetTrackerConfig
    selector: TargetSelectionManager

    def locate_by_contrast(
        self,
        frame: ProcessedFrame,
        predicted_bbox: BoundingBox,
        target_polarity: TargetPolarity,
        state: TrackerState,
        lost_frames: int,
        *,
        frame_degraded: bool,
    ) -> SearchResult | None:
        """Найти цель через локальный контраст около прогнозного bbox."""
        point = (int(predicted_bbox.center[0]), int(predicted_bbox.center[1]))
        selection = self.selector.apply(
            frame=frame,
            point=point,
            expected_bbox=predicted_bbox,
        )

        if selection is None:
            return None

        if selection.confidence < 0.7:
            return None

        if selection.polarity != target_polarity:
            return None

        candidate_bbox = selection.bbox.clamp(frame.bgr.shape)
        candidate_center = np.array(candidate_bbox.center, dtype=np.float32)
        predicted_center = np.array(predicted_bbox.center, dtype=np.float32)
        center_error = float(np.linalg.norm(candidate_center - predicted_center))
        max_dimension = max(predicted_bbox.width, predicted_bbox.height, self.config.min_box_size)

        if state == TrackerState.TRACKING:
            allowed_error = max(8.0, max_dimension * 0.95)
        else:
            allowed_error = max(12.0, max_dimension * (1.15 + 0.05 * min(lost_frames, 12)))

        if center_error > allowed_error:
            return None

        width_ratio = candidate_bbox.width / max(float(predicted_bbox.width), 1.0)
        height_ratio = candidate_bbox.height / max(float(predicted_bbox.height), 1.0)

        if max(width_ratio, height_ratio) > 1.8:
            return None

        if min(width_ratio, height_ratio) < 0.45:
            return None

        search_region = predicted_bbox.pad(
            self.config.search_margin + lost_frames * self.config.lost_search_growth,
            self.config.search_margin + lost_frames * self.config.lost_search_growth,
        ).clamp(frame.bgr.shape)

        if frame_degraded:
            score = max(
                self.config.template_update_threshold + 0.14,
                self.config.reacquire_threshold + 0.22,
            )
        else:
            score = min(
                self.config.template_update_threshold - 0.02,
                self.config.track_threshold + 0.18,
            )

        return SearchResult(
            bbox=candidate_bbox,
            score=score,
            search_region=search_region,
        )

    @staticmethod
    def resolve_target_polarity(
        frame: ProcessedFrame,
        bbox: BoundingBox,
        selection_polarity: TargetPolarity,
    ) -> TargetPolarity:
        """Определить полярность цели по selector или локальному фону."""
        if selection_polarity in (TargetPolarity.HOT, TargetPolarity.COLD):
            return selection_polarity

        gray = frame.normalized
        clamped = bbox.clamp(gray.shape)
        object_patch = gray[clamped.y:clamped.y2, clamped.x:clamped.x2]

        if object_patch.size == 0:
            return TargetPolarity.HOT

        margin = max(4, min(18, int(round(max(clamped.width, clamped.height) * 1.25))))
        outer = clamped.pad(margin, margin).clamp(gray.shape)
        ring_patch = gray[outer.y:outer.y2, outer.x:outer.x2]

        if ring_patch.size == 0:
            return TargetPolarity.HOT

        ring_mask = np.ones(ring_patch.shape, dtype=bool)
        inner_x1 = clamped.x - outer.x
        inner_y1 = clamped.y - outer.y
        inner_x2 = inner_x1 + clamped.width
        inner_y2 = inner_y1 + clamped.height
        ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        background_values = ring_patch[ring_mask]

        if background_values.size < 8:
            return TargetPolarity.HOT

        object_values = object_patch.reshape(-1)
        object_hot_level = float(np.percentile(object_values, 75))
        object_cold_level = float(np.percentile(object_values, 25))
        background_level = float(np.median(background_values))

        hot_contrast = object_hot_level - background_level
        cold_contrast = background_level - object_cold_level

        if hot_contrast >= cold_contrast:
            return TargetPolarity.HOT

        return TargetPolarity.COLD
