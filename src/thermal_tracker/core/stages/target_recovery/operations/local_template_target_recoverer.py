from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import cv2
import numpy as np

from ....config import PresetFieldReader
from ....domain.models import BoundingBox, ProcessedFrame
from ...frame_stabilization import FrameStabilizerResult
from ..result import TargetRecoveryResult
from ..type import TargetRecovererType
from .base_target_recoverer import BaseTargetRecoverer


class RecoveryImagePatch:
    """Работает с локальными patch для стадии восстановления цели."""

    @staticmethod
    def safe_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Изменить размер изображения без нулевой ширины или высоты."""
        width, height = size
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
        """Вырезать patch из изображения или вернуть None для вырожденной области."""
        clamped = bbox.clamp(image.shape)

        if clamped.width <= 1 or clamped.height <= 1:
            return None

        return image[clamped.y:clamped.y2, clamped.x:clamped.x2]


@dataclass(frozen=True, slots=True)
class LocalTemplateTargetRecovererConfig:
    """Хранит настройки восстановления цели по локальному template matching."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetRecovererType] = TargetRecovererType.LOCAL_TEMPLATE

    # Базовый отступ области поиска вокруг последнего bbox.
    search_padding: int = 60
    # Рост отступа области поиска за каждый потерянный кадр.
    search_padding_growth: int = 8
    # Масштабы шаблона при поиске цели.
    scales: tuple[float, ...] = (0.84, 0.94, 1.0, 1.08, 1.18)
    # Минимальный score template matching для принятия восстановления.
    match_threshold: float = 0.5
    # Скорость обновления адаптивного шаблона.
    template_alpha: float = 0.12

    def __post_init__(self) -> None:
        """Проверить корректность параметров template recoverer-а."""
        self._validate_non_negative_int(self.search_padding, "search_padding")
        self._validate_non_negative_int(self.search_padding_growth, "search_padding_growth")
        self._validate_positive_float_tuple(self.scales, "scales")
        self._validate_positive_float(self.match_threshold, "match_threshold")
        self._validate_ratio(self.template_alpha, "template_alpha")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_int_to(kwargs, "search_padding")
        reader.pop_int_to(kwargs, "search_padding_growth")
        reader.pop_float_tuple_to(kwargs, "scales")
        reader.pop_float_to(kwargs, "match_threshold")
        reader.pop_float_to(kwargs, "template_alpha")
        reader.ensure_empty()

        return cls(**kwargs)

    @staticmethod
    def _validate_non_negative_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")

    @staticmethod
    def _validate_positive_float(value: float, field_name: str) -> None:
        """Проверить, что вещественное значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

    @staticmethod
    def _validate_ratio(value: float, field_name: str) -> None:
        """Проверить, что значение находится в диапазоне [0, 1]."""
        if not 0 <= value <= 1:
            raise ValueError(f"{field_name} must be in range [0, 1].")

    @staticmethod
    def _validate_positive_float_tuple(value: tuple[float, ...], field_name: str) -> None:
        """Проверить, что кортеж содержит только положительные float-значения."""
        if not value:
            raise ValueError(f"{field_name} must not be empty.")

        if any(item <= 0 for item in value):
            raise ValueError(f"{field_name} must contain only positive values.")


@dataclass(slots=True)
class LocalTemplateTargetRecoverer(BaseTargetRecoverer):
    """
        Локальный recoverer на базе шаблонов цели.
        Стратегия повторяет принцип, который сейчас встроен в гибридный single-target
        tracker: ведутся два шаблона цели (долгий и адаптивный), при потере цель ищется
        локальной template-корреляцией в расширенной зоне поиска вокруг последнего
        известного bbox с перебором нескольких масштабов.
        Возвращается bbox с лучшим score, если он не ниже ``match_threshold``.
    """

    config: LocalTemplateTargetRecovererConfig

    _canonical_size: tuple[int, int] | None = None
    _long_term_template: np.ndarray | None = None
    _adaptive_template: np.ndarray | None = None

    def remember(self, frame: ProcessedFrame, bbox: BoundingBox) -> None:
        """Запомнить или адаптивно обновить шаблон уверенно найденной цели."""
        gray_patch = RecoveryImagePatch.crop(frame.gray, bbox)

        if gray_patch is None:
            return

        if not self._templates_ready:
            self._canonical_size = (gray_patch.shape[1], gray_patch.shape[0])
            self._long_term_template = gray_patch.copy()
            self._adaptive_template = gray_patch.copy()
            return

        assert self._canonical_size is not None
        assert self._adaptive_template is not None

        resized = RecoveryImagePatch.safe_resize(gray_patch, self._canonical_size)
        alpha = self.config.template_alpha
        adaptive_float = self._adaptive_template.astype(np.float32)
        new_float = resized.astype(np.float32)

        self._adaptive_template = cv2.addWeighted(
            new_float,
            alpha,
            adaptive_float,
            1.0 - alpha,
            0.0,
        ).astype(np.uint8)

    def recover(
        self,
        frame: ProcessedFrame,
        last_bbox: BoundingBox,
        motion: FrameStabilizerResult,
        lost_frames: int = 0,
    ) -> TargetRecoveryResult:
        """Найти цель по шаблону рядом с последним известным bbox."""
        if not self._templates_ready:
            return TargetRecoveryResult(
                bbox=None,
                source_name=str(self.config.operation_type),
                message="Local template recoverer has no remembered target.",
            )

        assert self._canonical_size is not None
        assert self._adaptive_template is not None
        assert self._long_term_template is not None

        frame_gray = frame.gray
        shifted_bbox = self._shift_bbox_by_motion(last_bbox, motion).clamp(frame_gray.shape)
        search_region = self._build_search_region(
            bbox=shifted_bbox,
            frame_shape=frame_gray.shape,
            lost_frames=lost_frames,
        )
        search_patch = RecoveryImagePatch.crop(frame_gray, search_region)

        if search_patch is None:
            return TargetRecoveryResult(
                bbox=None,
                search_region=search_region,
                source_name=str(self.config.operation_type),
                message="Search region is empty.",
            )

        best_bbox, best_score = self._find_best_template_match(
            frame_shape=frame_gray.shape,
            search_region=search_region,
            search_patch=search_patch,
        )

        if best_bbox is None or best_score < self.config.match_threshold:
            return TargetRecoveryResult(
                bbox=None,
                score=max(best_score, 0.0),
                search_region=search_region,
                source_name=str(self.config.operation_type),
                message="Target was not recovered by local template matching.",
            )

        return TargetRecoveryResult(
            bbox=best_bbox,
            score=best_score,
            search_region=search_region,
            source_name=str(self.config.operation_type),
            message="Target recovered by local template matching.",
        )

    def reset(self) -> None:
        """Забыть сохранённые шаблоны цели."""
        self._canonical_size = None
        self._long_term_template = None
        self._adaptive_template = None

    @property
    def _templates_ready(self) -> bool:
        """Проверить, готовы ли шаблоны для восстановления."""
        return (
            self._canonical_size is not None
            and self._long_term_template is not None
            and self._adaptive_template is not None
        )

    def _build_search_region(
        self,
        bbox: BoundingBox,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        lost_frames: int,
    ) -> BoundingBox:
        """Построить область поиска с учётом длительности потери."""
        effective_padding = (
            self.config.search_padding
            + max(0, int(lost_frames)) * self.config.search_padding_growth
        )

        return bbox.pad(effective_padding, effective_padding).clamp(frame_shape)

    def _find_best_template_match(
        self,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        search_region: BoundingBox,
        search_patch: np.ndarray,
    ) -> tuple[BoundingBox | None, float]:
        """Найти лучший bbox по адаптивному и долгому шаблонам."""
        assert self._canonical_size is not None
        assert self._adaptive_template is not None
        assert self._long_term_template is not None

        canonical_width, canonical_height = self._canonical_size
        best_score = -1.0
        best_bbox: BoundingBox | None = None

        for scale in self.config.scales:
            template_width = max(2, int(round(canonical_width * scale)))
            template_height = max(2, int(round(canonical_height * scale)))

            if template_width >= search_patch.shape[1] or template_height >= search_patch.shape[0]:
                continue

            adaptive_template = RecoveryImagePatch.safe_resize(
                self._adaptive_template,
                (template_width, template_height),
            )
            long_term_template = RecoveryImagePatch.safe_resize(
                self._long_term_template,
                (template_width, template_height),
            )
            score, location = self._best_match(
                search_patch=search_patch,
                adaptive_template=adaptive_template,
                long_term_template=long_term_template,
            )

            if score <= best_score:
                continue

            best_score = score
            best_bbox = BoundingBox(
                x=search_region.x + int(location[0]),
                y=search_region.y + int(location[1]),
                width=template_width,
                height=template_height,
            ).clamp(frame_shape)

        return best_bbox, best_score

    @staticmethod
    def _shift_bbox_by_motion(
        bbox: BoundingBox,
        motion: FrameStabilizerResult,
    ) -> BoundingBox:
        """Сместить bbox на глобальное движение камеры, если оно валидно."""
        if not motion.valid:
            return bbox

        center_x, center_y = bbox.center

        return BoundingBox.from_center(
            center_x + float(motion.dx),
            center_y + float(motion.dy),
            bbox.width,
            bbox.height,
        )

    @staticmethod
    def _best_match(
        search_patch: np.ndarray,
        adaptive_template: np.ndarray,
        long_term_template: np.ndarray,
    ) -> tuple[float, tuple[int, int]]:
        """Вернуть лучший score и позицию из двух шаблонов."""
        adaptive_response = cv2.matchTemplate(
            search_patch,
            adaptive_template,
            cv2.TM_CCOEFF_NORMED,
        )
        long_term_response = cv2.matchTemplate(
            search_patch,
            long_term_template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, adaptive_score, _, adaptive_location = cv2.minMaxLoc(adaptive_response)
        _, long_term_score, _, long_term_location = cv2.minMaxLoc(long_term_response)

        if float(adaptive_score) >= float(long_term_score):
            return float(adaptive_score), adaptive_location

        return float(long_term_score), long_term_location
