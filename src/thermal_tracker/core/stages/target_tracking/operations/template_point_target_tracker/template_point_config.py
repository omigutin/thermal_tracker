from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self

from ....target_selection.config import TargetSelectionConfig
from ....target_selection.operations import ContrastComponentTargetSelectorConfig
from ...type import TargetTrackerType
from .....config.preset_field_reader import PresetFieldReader


@dataclass(frozen=True, slots=True)
class TemplatePointTargetTrackerConfig:
    """Хранит настройки трекера цели по шаблонам и опорным точкам."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetTrackerType] = TargetTrackerType.TEMPLATE_POINT

    # Конфигурация выбора цели по первому клику и fallback-контрасту.
    target_selector_config: TargetSelectionConfig = field(
        default_factory=ContrastComponentTargetSelectorConfig,
    )

    # Минимальный размер bbox цели.
    min_box_size: int = 8
    # Базовый отступ области поиска вокруг прогноза.
    search_margin: int = 24
    # Узкий отступ области поиска при хорошем прогнозе по точкам.
    point_search_margin: int = 14
    # Рост области поиска за каждый потерянный кадр.
    lost_search_growth: int = 18
    # Количество потерянных кадров, после которого разрешён поиск по всему кадру.
    full_frame_after: int = 999
    # Максимальное количество потерянных кадров перед сбросом трека.
    max_lost_frames: int = 70

    # Порог уверенного template tracking.
    track_threshold: float = 0.42
    # Порог повторного захвата после потери.
    reacquire_threshold: float = 0.5
    # Порог, после которого можно обновлять адаптивный шаблон.
    template_update_threshold: float = 0.62
    # Коэффициенты масштабирования bbox при template search.
    scales: tuple[float, ...] = (0.72, 0.84, 0.94, 1.0, 1.08, 1.18, 1.32)
    # Штраф за удалённость кандидата от прогноза.
    distance_penalty: float = 0.14

    # Скорость обновления адаптивного шаблона.
    template_alpha: float = 0.12
    # Сглаживание остаточной скорости цели относительно движения камеры.
    velocity_alpha: float = 0.45

    # Максимальный сдвиг центра при обычном сопровождении.
    max_tracking_center_shift: float = 1.6
    # Максимальный сдвиг центра при повторном захвате.
    max_reacquire_center_shift: float = 2.8
    # Рост допустимого сдвига центра при долгой потере.
    reacquire_center_growth: float = 0.32

    # Минимальное сжатие bbox между соседними измерениями.
    max_size_shrink: float = 0.72
    # Максимальный рост bbox между соседними измерениями.
    max_size_growth: float = 1.25
    # Минимальное сжатие bbox при повторном захвате.
    max_size_shrink_on_reacquire: float = 0.55
    # Максимальный рост bbox при повторном захвате.
    max_size_growth_on_reacquire: float = 1.6
    # Максимальный рост bbox относительно стартового размера.
    max_size_growth_from_initial: float = 4.0

    # Максимальное количество кадров потери после контакта цели с краем кадра.
    edge_exit_max_lost_frames: int = 4
    # Отступ от края кадра для фиксации возможного выхода цели.
    edge_exit_margin: int = 10

    # Максимальное количество опорных точек на цели.
    max_feature_points: int = 40
    # Минимальное количество опорных точек для использования прогноза.
    min_feature_points: int = 6
    # Минимальное качество точки для cv2.goodFeaturesToTrack.
    feature_quality_level: float = 0.02
    # Минимальная дистанция между опорными точками.
    feature_min_distance: int = 6
    # Интервал принудительного обновления опорных точек.
    feature_refresh_interval: int = 6

    # Размер окна LK optical flow.
    optical_flow_window_size: int = 21
    # Количество уровней пирамиды LK optical flow.
    optical_flow_max_level: int = 3
    # Максимальное число итераций LK optical flow.
    optical_flow_criteria_count: int = 20
    # Точность остановки LK optical flow.
    optical_flow_criteria_eps: float = 0.03
    # Максимальная ошибка LK optical flow для принятия точки.
    optical_flow_max_error: float = 25.0

    # Включает удержание прогноза при деградации кадра.
    blur_hold_enabled: bool = True
    # Доля падения резкости, после которой кадр считается деградированным.
    blur_sharpness_drop_ratio: float = 0.5
    # Максимальное количество кадров удержания после blur.
    blur_hold_max_frames: int = 120
    # Рост допустимого отклонения центра при blur hold.
    blur_hold_center_growth: float = 0.06
    # Скорость адаптации базовой резкости.
    sharpness_baseline_alpha: float = 0.06
    # Минимальное значение baseline резкости.
    min_sharpness_baseline: float = 1e-6

    # Нижняя граница центральной ROI по высоте для оценки резкости.
    sharpness_roi_y_min: float = 0.18
    # Верхняя граница центральной ROI по высоте для оценки резкости.
    sharpness_roi_y_max: float = 0.82
    # Левая граница центральной ROI по ширине для оценки резкости.
    sharpness_roi_x_min: float = 0.08
    # Правая граница центральной ROI по ширине для оценки резкости.
    sharpness_roi_x_max: float = 0.92
    # Размер ядра Лапласиана для оценки резкости.
    sharpness_laplacian_kernel: int = 3
    # Перцентиль Лапласиана для оценки резкости.
    sharpness_percentile: float = 90.0

    def __post_init__(self) -> None:
        """Проверить корректность параметров template-point трекера."""
        self._validate_positive_int(self.min_box_size, "min_box_size")
        self._validate_non_negative_int(self.search_margin, "search_margin")
        self._validate_non_negative_int(self.point_search_margin, "point_search_margin")
        self._validate_non_negative_int(self.lost_search_growth, "lost_search_growth")
        self._validate_non_negative_int(self.full_frame_after, "full_frame_after")
        self._validate_non_negative_int(self.max_lost_frames, "max_lost_frames")
        self._validate_positive_float(self.track_threshold, "track_threshold")
        self._validate_positive_float(self.reacquire_threshold, "reacquire_threshold")
        self._validate_positive_float(self.template_update_threshold, "template_update_threshold")
        self._validate_positive_float_tuple(self.scales, "scales")
        self._validate_non_negative_float(self.distance_penalty, "distance_penalty")
        self._validate_ratio(self.template_alpha, "template_alpha")
        self._validate_ratio(self.velocity_alpha, "velocity_alpha")
        self._validate_positive_int(self.max_feature_points, "max_feature_points")
        self._validate_positive_int(self.min_feature_points, "min_feature_points")
        self._validate_ratio(self.feature_quality_level, "feature_quality_level")
        self._validate_positive_int(self.feature_min_distance, "feature_min_distance")
        self._validate_positive_int(self.feature_refresh_interval, "feature_refresh_interval")
        self._validate_positive_int(self.optical_flow_window_size, "optical_flow_window_size")
        self._validate_non_negative_int(self.optical_flow_max_level, "optical_flow_max_level")
        self._validate_positive_int(self.optical_flow_criteria_count, "optical_flow_criteria_count")
        self._validate_positive_float(self.optical_flow_criteria_eps, "optical_flow_criteria_eps")
        self._validate_positive_float(self.optical_flow_max_error, "optical_flow_max_error")
        self._validate_ratio(self.blur_sharpness_drop_ratio, "blur_sharpness_drop_ratio")
        self._validate_non_negative_int(self.blur_hold_max_frames, "blur_hold_max_frames")
        self._validate_non_negative_float(self.blur_hold_center_growth, "blur_hold_center_growth")
        self._validate_ratio(self.sharpness_baseline_alpha, "sharpness_baseline_alpha")
        self._validate_positive_float(self.min_sharpness_baseline, "min_sharpness_baseline")
        self._validate_ratio(self.sharpness_roi_y_min, "sharpness_roi_y_min")
        self._validate_ratio(self.sharpness_roi_y_max, "sharpness_roi_y_max")
        self._validate_ratio(self.sharpness_roi_x_min, "sharpness_roi_x_min")
        self._validate_ratio(self.sharpness_roi_x_max, "sharpness_roi_x_max")
        self._validate_sobel_like_kernel(self.sharpness_laplacian_kernel, "sharpness_laplacian_kernel")

        if self.min_feature_points > self.max_feature_points:
            raise ValueError("min_feature_points must be less than or equal to max_feature_points.")

        if self.sharpness_roi_y_min >= self.sharpness_roi_y_max:
            raise ValueError("sharpness_roi_y_min must be less than sharpness_roi_y_max.")

        if self.sharpness_roi_x_min >= self.sharpness_roi_x_max:
            raise ValueError("sharpness_roi_x_min must be less than sharpness_roi_x_max.")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        reader.pop_bool_to(kwargs, "enabled")
        reader.pop_bool_to(kwargs, "blur_hold_enabled")

        for field_name in (
            "min_box_size",
            "search_margin",
            "point_search_margin",
            "lost_search_growth",
            "full_frame_after",
            "max_lost_frames",
            "edge_exit_max_lost_frames",
            "edge_exit_margin",
            "max_feature_points",
            "min_feature_points",
            "feature_min_distance",
            "feature_refresh_interval",
            "optical_flow_window_size",
            "optical_flow_max_level",
            "optical_flow_criteria_count",
            "blur_hold_max_frames",
            "sharpness_laplacian_kernel",
        ):
            reader.pop_int_to(kwargs, field_name)

        for field_name in (
            "track_threshold",
            "reacquire_threshold",
            "template_update_threshold",
            "distance_penalty",
            "template_alpha",
            "velocity_alpha",
            "max_tracking_center_shift",
            "max_reacquire_center_shift",
            "reacquire_center_growth",
            "max_size_shrink",
            "max_size_growth",
            "max_size_shrink_on_reacquire",
            "max_size_growth_on_reacquire",
            "max_size_growth_from_initial",
            "feature_quality_level",
            "optical_flow_criteria_eps",
            "optical_flow_max_error",
            "blur_sharpness_drop_ratio",
            "blur_hold_center_growth",
            "sharpness_baseline_alpha",
            "min_sharpness_baseline",
            "sharpness_roi_y_min",
            "sharpness_roi_y_max",
            "sharpness_roi_x_min",
            "sharpness_roi_x_max",
            "sharpness_percentile",
        ):
            reader.pop_float_to(kwargs, field_name)

        reader.pop_float_tuple_to(kwargs, "scales")
        reader.ensure_empty()

        return cls(**kwargs)

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

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
    def _validate_non_negative_float(value: float, field_name: str) -> None:
        """Проверить, что вещественное значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")

    @staticmethod
    def _validate_ratio(value: float, field_name: str) -> None:
        """Проверить, что значение находится в диапазоне (0, 1]."""
        if not 0 < value <= 1:
            raise ValueError(f"{field_name} must be in range (0, 1].")

    @staticmethod
    def _validate_positive_float_tuple(value: tuple[float, ...], field_name: str) -> None:
        """Проверить, что кортеж float содержит только положительные значения."""
        if not value:
            raise ValueError(f"{field_name} must not be empty.")

        if any(item <= 0 for item in value):
            raise ValueError(f"{field_name} must contain only positive values.")

    @staticmethod
    def _validate_sobel_like_kernel(value: int, field_name: str) -> None:
        """Проверить размер ядра OpenCV-фильтра."""
        if value not in (1, 3, 5, 7):
            raise ValueError(f"{field_name} must be one of: 1, 3, 5, 7.")
