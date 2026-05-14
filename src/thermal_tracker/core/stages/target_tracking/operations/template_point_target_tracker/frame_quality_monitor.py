from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .....domain.models import ProcessedFrame
from .template_point_config import TemplatePointTargetTrackerConfig


@dataclass(slots=True)
class FrameQualityMonitor:
    """Отслеживает деградацию кадра и режим удержания прогноза."""

    config: TemplatePointTargetTrackerConfig
    sharpness_baseline: float | None = field(default=None, init=False)
    degraded_frames: int = field(default=0, init=False)
    blur_hold_frames: int = field(default=0, init=False)

    def reset(self) -> None:
        """Сбросить состояние контроля качества кадра."""
        self.sharpness_baseline = None
        self.degraded_frames = 0
        self.blur_hold_frames = 0

    def update_state(self, frame: ProcessedFrame) -> bool:
        """Обновить состояние качества кадра и вернуть признак деградации."""
        if not self.config.blur_hold_enabled:
            self.degraded_frames = 0
            self.blur_hold_frames = 0
            return False

        sharpness = self.measure_sharpness(frame)

        if self.sharpness_baseline is None:
            self.sharpness_baseline = max(sharpness, self.config.min_sharpness_baseline)
            return False

        baseline = max(self.sharpness_baseline, self.config.min_sharpness_baseline)
        degraded = sharpness <= baseline * self.config.blur_sharpness_drop_ratio

        if degraded:
            self.degraded_frames += 1
            self.blur_hold_frames = max(self.blur_hold_frames, self.config.blur_hold_max_frames)
        else:
            self.degraded_frames = 0
            if self.blur_hold_frames > 0:
                self.blur_hold_frames -= 1

        return degraded

    def update_baseline(self, frame: ProcessedFrame) -> None:
        """Медленно обновить базовую резкость по хорошему кадру."""
        sharpness = self.measure_sharpness(frame)

        if self.sharpness_baseline is None:
            self.sharpness_baseline = max(sharpness, self.config.min_sharpness_baseline)
            return

        alpha = self.config.sharpness_baseline_alpha
        self.sharpness_baseline = self.sharpness_baseline * (1.0 - alpha) + sharpness * alpha

    def blur_hold_active(self) -> bool:
        """Проверить, активен ли режим удержания после blur."""
        return (
            self.config.blur_hold_enabled
            and (self.degraded_frames > 0 or self.blur_hold_frames > 0)
        )

    def current_max_lost_frames(self) -> int:
        """Вернуть текущий лимит потерянных кадров с учётом blur hold."""
        if not self.blur_hold_active():
            return self.config.max_lost_frames

        return self.config.max_lost_frames + self.config.blur_hold_max_frames

    def measure_sharpness(self, frame: ProcessedFrame) -> float:
        """Оценить резкость центральной части кадра через Лапласиан."""
        gray = frame.gray
        frame_height, frame_width = gray.shape[:2]

        x1 = int(round(frame_width * self.config.sharpness_roi_x_min))
        x2 = int(round(frame_width * self.config.sharpness_roi_x_max))
        y1 = int(round(frame_height * self.config.sharpness_roi_y_min))
        y2 = int(round(frame_height * self.config.sharpness_roi_y_max))

        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            roi = gray

        laplacian = cv2.Laplacian(
            roi,
            cv2.CV_32F,
            ksize=self.config.sharpness_laplacian_kernel,
        )

        return float(np.percentile(np.abs(laplacian), self.config.sharpness_percentile))
    