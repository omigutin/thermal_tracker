from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .....domain.models import BoundingBox, ProcessedFrame
from .image_patch import ImagePatch
from .template_point_config import TemplatePointTargetTrackerConfig


@dataclass(slots=True)
class TemplateStorage:
    """Хранит долгий и адаптивный шаблоны выбранной цели."""

    config: TemplatePointTargetTrackerConfig
    canonical_size: tuple[int, int] | None = field(default=None, init=False)
    long_term_gray: np.ndarray | None = field(default=None, init=False, repr=False)
    long_term_grad: np.ndarray | None = field(default=None, init=False, repr=False)
    adaptive_gray: np.ndarray | None = field(default=None, init=False, repr=False)
    adaptive_grad: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def is_ready(self) -> bool:
        """Проверить, готовы ли шаблоны к поиску цели."""
        return (
            self.canonical_size is not None
            and self.long_term_gray is not None
            and self.long_term_grad is not None
            and self.adaptive_gray is not None
            and self.adaptive_grad is not None
        )

    def reset(self) -> None:
        """Сбросить сохранённые шаблоны."""
        self.canonical_size = None
        self.long_term_gray = None
        self.long_term_grad = None
        self.adaptive_gray = None
        self.adaptive_grad = None

    def initialize(self, frame: ProcessedFrame, bbox: BoundingBox) -> bool:
        """Создать долгий и адаптивный шаблоны по стартовому bbox."""
        canonical_width = max(self.config.min_box_size, bbox.width)
        canonical_height = max(self.config.min_box_size, bbox.height)
        canonical_size = (canonical_width, canonical_height)

        gray_patch = ImagePatch.crop(frame.normalized, bbox)
        grad_patch = ImagePatch.crop(frame.gradient, bbox)

        if gray_patch is None or grad_patch is None:
            return False

        self.canonical_size = canonical_size
        self.long_term_gray = ImagePatch.safe_resize(gray_patch, canonical_size)
        self.long_term_grad = ImagePatch.safe_resize(grad_patch, canonical_size)
        self.adaptive_gray = self.long_term_gray.copy()
        self.adaptive_grad = self.long_term_grad.copy()

        return True

    def can_update(self, bbox: BoundingBox, score: float) -> bool:
        """Проверить, можно ли обновлять адаптивный шаблон текущим bbox."""
        if score < self.config.template_update_threshold:
            return False

        if self.canonical_size is None:
            return True

        canonical_width, canonical_height = self.canonical_size
        width_ratio = bbox.width / max(float(canonical_width), 1.0)
        height_ratio = bbox.height / max(float(canonical_height), 1.0)

        return not (
            max(width_ratio, height_ratio) > 1.28
            and score < self.config.template_update_threshold + 0.18
        )

    def update(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Аккуратно обновить адаптивный шаблон уверенным измерением."""
        if self.canonical_size is None:
            return
        if self.adaptive_gray is None or self.adaptive_grad is None:
            return

        gray_patch = ImagePatch.crop(frame.normalized, bbox)
        grad_patch = ImagePatch.crop(frame.gradient, bbox)

        if gray_patch is None or grad_patch is None:
            return

        new_gray = ImagePatch.safe_resize(gray_patch, self.canonical_size).astype(np.float32)
        new_grad = ImagePatch.safe_resize(grad_patch, self.canonical_size).astype(np.float32)

        alpha = self.config.template_alpha
        adaptive_gray = self.adaptive_gray.astype(np.float32)
        adaptive_grad = self.adaptive_grad.astype(np.float32)

        self.adaptive_gray = cv2.addWeighted(
            new_gray,
            alpha,
            adaptive_gray,
            1.0 - alpha,
            0.0,
        ).astype(np.uint8)
        self.adaptive_grad = cv2.addWeighted(
            new_grad,
            alpha,
            adaptive_grad,
            1.0 - alpha,
            0.0,
        ).astype(np.uint8)
        