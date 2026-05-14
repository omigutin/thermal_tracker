from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .....domain.models import BoundingBox, ProcessedFrame, TrackerState
from ....target_selection import TargetPolarity
from .search_result import SearchResult
from .image_patch import ImagePatch
from .point_prediction import PointPrediction
from .template_point_config import TemplatePointTargetTrackerConfig
from .template_storage import TemplateStorage


@dataclass(slots=True)
class TemplateMatcher:
    """Ищет цель по долгому и адаптивному шаблону."""

    config: TemplatePointTargetTrackerConfig
    storage: TemplateStorage

    def locate_target(
        self,
        frame: ProcessedFrame,
        predicted_bbox: BoundingBox,
        search_region: BoundingBox,
        point_prediction: PointPrediction | None,
        target_polarity: TargetPolarity,
        state: TrackerState,
        lost_frames: int,
    ) -> SearchResult | None:
        """Найти лучший bbox цели в области поиска."""
        if not self.storage.is_ready:
            return None

        assert self.storage.adaptive_gray is not None
        assert self.storage.long_term_gray is not None

        search_gray = ImagePatch.crop(frame.normalized, search_region)

        if search_gray is None:
            return None

        best_result: SearchResult | None = None

        for scale in self.config.scales:
            candidate_width = max(self.config.min_box_size, int(round(predicted_bbox.width * scale)))
            candidate_height = max(self.config.min_box_size, int(round(predicted_bbox.height * scale)))

            if candidate_width >= search_region.width or candidate_height >= search_region.height:
                continue

            adaptive_template = ImagePatch.safe_resize(
                self.storage.adaptive_gray,
                (candidate_width, candidate_height),
            )
            long_term_template = ImagePatch.safe_resize(
                self.storage.long_term_gray,
                (candidate_width, candidate_height),
            )
            candidate = self._evaluate_template_pair(
                frame=frame,
                search_region=search_region,
                search_gray=search_gray,
                template_gray=adaptive_template,
                fallback_template=long_term_template,
                candidate_width=candidate_width,
                candidate_height=candidate_height,
                predicted_bbox=predicted_bbox,
                point_prediction=point_prediction,
                target_polarity=target_polarity,
                state=state,
                lost_frames=lost_frames,
            )

            if candidate is None:
                continue

            if best_result is None or candidate.score > best_result.score:
                best_result = candidate

        return best_result

    def _evaluate_template_pair(
        self,
        frame: ProcessedFrame,
        search_region: BoundingBox,
        search_gray: np.ndarray,
        template_gray: np.ndarray,
        fallback_template: np.ndarray,
        candidate_width: int,
        candidate_height: int,
        predicted_bbox: BoundingBox,
        point_prediction: PointPrediction | None,
        target_polarity: TargetPolarity,
        state: TrackerState,
        lost_frames: int,
    ) -> SearchResult | None:
        """Проверить кандидатов по адаптивному и долгому шаблону."""
        response_primary = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        response_fallback = cv2.matchTemplate(search_gray, fallback_template, cv2.TM_CCOEFF_NORMED)

        candidates: list[tuple[float, tuple[int, int]]] = []

        for response in (response_primary, response_fallback):
            candidates.extend(
                self._collect_template_candidates(
                    response=response,
                    candidate_width=candidate_width,
                    candidate_height=candidate_height,
                )
            )

        best_score = -1.0
        best_bbox: BoundingBox | None = None

        for template_score, max_location in candidates:
            candidate_bbox = BoundingBox(
                x=search_region.x + int(max_location[0]),
                y=search_region.y + int(max_location[1]),
                width=candidate_width,
                height=candidate_height,
            ).clamp(frame.bgr.shape)

            score = self._score_candidate(
                frame=frame,
                candidate_bbox=candidate_bbox,
                predicted_bbox=predicted_bbox,
                template_score=template_score,
                search_region=search_region,
                point_prediction=point_prediction,
                target_polarity=target_polarity,
                state=state,
                lost_frames=lost_frames,
            )

            if score > best_score:
                best_score = score
                best_bbox = candidate_bbox

        if best_bbox is None:
            return None

        return SearchResult(bbox=best_bbox, score=best_score, search_region=search_region)

    def _collect_template_candidates(
        self,
        response: np.ndarray,
        candidate_width: int,
        candidate_height: int,
        limit: int = 8,
    ) -> list[tuple[float, tuple[int, int]]]:
        """Собрать несколько локальных максимумов template response."""
        work = response.copy()
        candidates: list[tuple[float, tuple[int, int]]] = []
        suppression_radius = max(3, min(candidate_width, candidate_height) // 2)

        for _ in range(limit):
            _, max_value, _, max_location = cv2.minMaxLoc(work)

            if float(max_value) <= 0.05:
                break

            candidates.append((float(max_value), max_location))

            x, y = max_location
            left = max(0, x - suppression_radius)
            right = min(work.shape[1], x + suppression_radius + 1)
            top = max(0, y - suppression_radius)
            bottom = min(work.shape[0], y + suppression_radius + 1)
            work[top:bottom, left:right] = -1.0

        return candidates

    def _score_candidate(
        self,
        frame: ProcessedFrame,
        candidate_bbox: BoundingBox,
        predicted_bbox: BoundingBox,
        template_score: float,
        search_region: BoundingBox,
        point_prediction: PointPrediction | None,
        target_polarity: TargetPolarity,
        state: TrackerState,
        lost_frames: int,
    ) -> float:
        """Рассчитать итоговый score кандидата цели."""
        if not self.storage.is_ready:
            return -1.0

        assert self.storage.canonical_size is not None
        assert self.storage.adaptive_gray is not None
        assert self.storage.adaptive_grad is not None
        assert self.storage.long_term_gray is not None
        assert self.storage.long_term_grad is not None

        candidate_gray = ImagePatch.crop(frame.normalized, candidate_bbox)
        candidate_grad = ImagePatch.crop(frame.gradient, candidate_bbox)

        if candidate_gray is None or candidate_grad is None:
            return -1.0

        normalized_gray = ImagePatch.safe_resize(candidate_gray, self.storage.canonical_size)
        normalized_grad = ImagePatch.safe_resize(candidate_grad, self.storage.canonical_size)

        gray_adaptive = ImagePatch.correlation(normalized_gray, self.storage.adaptive_gray)
        gray_long = ImagePatch.correlation(normalized_gray, self.storage.long_term_gray)
        grad_adaptive = ImagePatch.correlation(normalized_grad, self.storage.adaptive_grad)
        grad_long = ImagePatch.correlation(normalized_grad, self.storage.long_term_grad)
        contrast_score = self._local_contrast_score(frame, candidate_bbox, target_polarity)

        predicted_center = np.array(predicted_bbox.center, dtype=np.float32)
        candidate_center = np.array(candidate_bbox.center, dtype=np.float32)
        center_distance = float(np.linalg.norm(candidate_center - predicted_center))
        search_scale = max(search_region.width, search_region.height, 1)
        distance_penalty = (center_distance / search_scale) * self.config.distance_penalty
        max_dimension = max(predicted_bbox.width, predicted_bbox.height, self.config.min_box_size)

        if point_prediction is not None:
            point_center = np.array(point_prediction.bbox.center, dtype=np.float32)
            point_distance = float(np.linalg.norm(candidate_center - point_center))
            point_penalty = (
                (point_distance / search_scale)
                * self.config.distance_penalty
                * (0.8 + point_prediction.confidence)
            )

            if lost_frames == 0 and point_distance > max(predicted_bbox.width, predicted_bbox.height) * 1.4:
                return -1.0
        else:
            point_penalty = 0.0

            if lost_frames == 0:
                max_center_distance = max_dimension * self.config.max_tracking_center_shift
            else:
                max_center_distance = max_dimension * (
                    self.config.max_reacquire_center_shift
                    + self.config.reacquire_center_growth * min(lost_frames, 8)
                )

            if center_distance > max_center_distance and lost_frames < self.config.full_frame_after:
                return -1.0

        combined = (
            template_score * 0.24
            + gray_adaptive * 0.27
            + gray_long * 0.16
            + grad_adaptive * 0.14
            + grad_long * 0.07
            + contrast_score * 0.12
        )

        return combined - distance_penalty - point_penalty

    def _local_contrast_score(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
        target_polarity: TargetPolarity,
    ) -> float:
        """Оценить, насколько кандидат отделяется от локального фона."""
        levels = self._measure_local_contrast_levels(frame=frame, bbox=bbox)

        if levels is None:
            return -0.2

        object_hot_level, object_cold_level, background_level = levels

        if target_polarity == TargetPolarity.COLD:
            contrast = background_level - object_cold_level
        else:
            contrast = object_hot_level - background_level

        normalized = (contrast - 4.0) / 32.0

        return float(np.clip(normalized, -1.0, 1.0))

    @staticmethod
    def _measure_local_contrast_levels(
        frame: ProcessedFrame,
        bbox: BoundingBox,
    ) -> tuple[float, float, float] | None:
        """Измерить уровни яркости объекта и фонового кольца."""
        image = frame.normalized
        clamped = bbox.clamp(image.shape)
        object_patch = image[clamped.y:clamped.y2, clamped.x:clamped.x2]

        if object_patch.size == 0:
            return None

        margin = max(4, min(18, int(round(max(clamped.width, clamped.height) * 1.25))))
        outer = clamped.pad(margin, margin).clamp(image.shape)
        ring_patch = image[outer.y:outer.y2, outer.x:outer.x2]

        if ring_patch.size == 0:
            return None

        ring_mask = np.ones(ring_patch.shape, dtype=bool)
        inner_x1 = clamped.x - outer.x
        inner_y1 = clamped.y - outer.y
        inner_x2 = inner_x1 + clamped.width
        inner_y2 = inner_y1 + clamped.height
        ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        background_values = ring_patch[ring_mask]

        if background_values.size < 8:
            return None

        object_values = object_patch.reshape(-1)
        object_hot_level = float(np.percentile(object_values, 75))
        object_cold_level = float(np.percentile(object_values, 25))
        background_level = float(np.median(background_values))

        return object_hot_level, object_cold_level, background_level
