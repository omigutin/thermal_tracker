from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Self

from .....config.preset_field_reader import PresetFieldReader
from .....domain.models import BoundingBox, ProcessedFrame
from ...result import TargetPolarity, TargetSelectorResult
from ...type import TargetSelectorType
from ..base_target_selector import BaseTargetSelector
from .contrast_component_extractor import ContrastComponentExtractor
from .contrast_component_mask_builder import ContrastComponentMaskBuilder
from .contrast_component_patch_extractor import ContrastComponentPatchExtractor
from .contrast_component_selection_expander import ClickSelectionExpander


@dataclass(frozen=True, slots=True)
class ContrastComponentTargetSelectorConfig:
    """Хранит настройки выбора цели по клику."""

    # Включает или отключает операцию.
    enabled: bool = True
    # Тип операции для связи конфигурации с фабрикой.
    operation_type: ClassVar[TargetSelectorType] = TargetSelectorType.CONTRAST_COMPONENT

    # Базовый радиус локального участка вокруг точки клика.
    search_radius: int = 32
    # Максимальный радиус повторной попытки выбора цели.
    max_retry_radius: int = 96
    # Множитель радиуса для повторной попытки выбора цели.
    retry_scale: float = 1.8
    # Отступ вокруг найденной области.
    padding: int = 2
    # Размер fallback-bbox, если нормальная сегментация не сработала.
    fallback_size: int = 16

    # Минимальная площадь компоненты, которую можно считать целью.
    min_component_area: int = 24
    # Максимальная доля площади patch, которую может занимать компонента.
    max_component_fill: float = 0.55
    # Максимальная доля ширины/высоты patch, которую может занимать bbox компоненты.
    max_patch_span_ratio: float = 0.85
    # Максимальный рост bbox при уточнении уже выбранной цели.
    max_refine_growth: float = 1.8

    # Радиус локального окна вокруг клика для оценки шума и разброса яркости.
    local_window_radius: int = 3
    # Коэффициент, который превращает локальное стандартное отклонение в допуск яркости.
    similarity_sigma: float = 2.0
    # Минимальный допустимый порог похожести по яркости.
    min_tolerance: int = 8
    # Максимальный допустимый порог похожести по яркости.
    max_tolerance: int = 42
    # Минимальный контраст объекта относительно окружения.
    min_object_contrast: float = 8.0

    # Размер ядра морфологической очистки бинарной маски.
    mask_morphology_kernel: int = 3
    # Количество итераций удаления мелкого шума из маски.
    mask_open_iterations: int = 1
    # Количество итераций заполнения небольших разрывов внутри маски.
    mask_close_iterations: int = 2

    # Размер ядра размытия для резервной score-маски.
    score_blur_kernel: int = 5
    # Размер ядра оператора Sobel для оценки локальных границ.
    score_sobel_kernel: int = 3
    # Вес похожести пикселя на кликнутую яркость в резервной score-маске.
    score_difference_weight: float = 0.6
    # Вес локального градиента в резервной score-маске.
    score_gradient_weight: float = 0.3
    # Вес расстояния от клика в резервной score-маске.
    score_distance_weight: float = 0.1
    # Максимальное значение score, при котором пиксель попадает в резервную маску.
    score_threshold: float = 0.95

    # Минимальный радиус окна для смещения клика к более выразительному пикселю.
    snap_min_radius: int = 6
    # Множитель радиуса локального окна для поиска более выразительного пикселя.
    snap_radius_multiplier: float = 2.0
    # Дополнительный запас радиуса при поиске более выразительного пикселя.
    snap_radius_padding: int = 2
    # Размер ядра размытия перед оценкой выразительности пикселей.
    snap_blur_kernel: int = 5
    # Штраф за удалённость пикселя от исходной точки клика.
    snap_distance_weight: float = 1.6

    # Доля размера ожидаемого bbox для расчёта радиуса уточняющего patch.
    refine_patch_bbox_scale: float = 0.8
    # Минимальная доля базового радиуса при уточнении уже известной цели.
    refine_patch_min_radius_ratio: float = 0.5

    # Толщина фонового кольца вокруг компоненты для оценки контраста.
    background_ring: int = 4
    # Минимальное количество фоновых пикселей для оценки контраста.
    min_background_pixels: int = 10
    # Уверенность для компактного fallback-выбора.
    compact_selection_confidence: float = 0.45
    # Минимальный радиус компактного выбора вокруг клика.
    compact_min_radius: int = 5
    # Максимальный радиус компактного выбора вокруг клика.
    compact_max_radius: int = 14

    # Размер ядра размытия для контрастной компоненты.
    contrast_blur_kernel: int = 5
    # Размер ядра морфологической очистки контрастной маски.
    contrast_morphology_kernel: int = 3
    # Количество итераций удаления шума из контрастной маски.
    contrast_open_iterations: int = 1
    # Количество итераций закрытия разрывов в контрастной маске.
    contrast_close_iterations: int = 1

    # Минимальная площадь bbox, с которой можно пытаться делить контрастный кластер.
    contrast_split_min_bbox_area: int = 1800
    # Минимальная сторона bbox, с которой можно пытаться делить контрастный кластер.
    contrast_split_min_bbox_side: int = 55
    # Максимальная доля patch, которую может занимать bbox для деления контрастного кластера.
    contrast_split_max_patch_fill: float = 0.25
    # Во сколько раз компонент должен быть больше минимальной площади для деления.
    contrast_split_min_area_multiplier: int = 4
    # Квантили, которыми пробуем выделять ядра внутри контрастного кластера.
    contrast_split_quantiles: tuple[float, ...] = (0.55, 0.65, 0.75)
    # Минимальная площадь ядра при делении контрастного кластера.
    contrast_core_min_area: int = 6
    # Доля min_component_area для минимальной площади ядра.
    contrast_core_min_area_ratio: float = 0.5
    # Размер ядра расширения выбранного ядра контрастного кластера.
    contrast_core_dilate_kernel: int = 3
    # Количество итераций расширения выбранного ядра контрастного кластера.
    contrast_core_dilate_iterations: int = 1
    # Максимальная доля исходного bbox, которую может занимать split-bbox.
    contrast_split_max_bbox_ratio: float = 0.78

    # Во сколько раз крупная компонента должна превышать min_component_area для деления.
    large_component_min_area_multiplier: int = 4
    # Размеры ядер эрозии для попыток разделить крупную компоненту.
    split_kernel_sizes: tuple[int, ...] = (3, 5, 7, 9, 11)
    # Минимальная доля исходной площади для принятия split-кандидата.
    split_min_area_ratio: float = 0.08
    # Требуемое улучшение площади split-кандидата относительно текущего лучшего.
    split_improvement_ratio: float = 0.88

    # Минимальная заполненность patch для повторного ужесточения крупной компоненты.
    tighten_min_area_fill: float = 0.16
    # Минимальная доля span компоненты для повторного ужесточения.
    tighten_min_span_ratio: float = 0.62
    # Масштабы tolerance для повторного ужесточения крупной компоненты.
    tighten_tolerance_scales: tuple[float, ...] = (0.55, 0.4, 0.3)
    # Минимальный множитель площади при повторном ужесточении.
    tighten_min_area_multiplier: int = 3
    # Максимальная доля исходного bbox для принятия ужатой компоненты.
    tighten_max_bbox_area_ratio: float = 0.78

    # Максимальный fallback_size, при котором включается режим маленькой цели.
    small_target_max_fallback_size: int = 18
    # Максимальный max_expansion_ratio, при котором включается режим маленькой цели.
    small_target_max_expansion_ratio: float = 2.5
    # Множитель стороны ожидаемой маленькой цели для признака oversized.
    oversized_side_multiplier: float = 2.0
    # Множитель площади ожидаемой маленькой цели для признака oversized.
    oversized_area_multiplier: float = 4.0
    # Aspect ratio, после которого компонент считается вытянутым.
    elongated_aspect_ratio: float = 1.75
    # Доля search_radius, после которой длинный компонент считается линией.
    long_line_search_radius_ratio: float = 0.75
    # Множитель площади ожидаемой цели для отсечения огромных компонент.
    oversized_area_min_multiplier: int = 24
    # Минимальная абсолютная площадь для отсечения огромных компонент.
    oversized_area_min_pixels: int = 260

    # Минимальный внешний отступ вокруг seed-области для поиска полного объекта.
    expansion_margin: int = 4
    # Множитель размера seed-области для расчёта рабочего отступа.
    expansion_margin_seed_scale: float = 0.5
    # Максимальная площадь seed-области, которую имеет смысл расширять.
    expansion_seed_max_area: int = 3000
    # Максимальная доля рабочего patch, которую может занять расширенная область.
    max_expanded_fill: float = 0.65
    # Доля контраста между объектом и фоном для построения порога foreground.
    foreground_fraction: float = 0.5

    # Размер ядра морфологической очистки расширенной маски.
    expansion_morphology_kernel: int = 5
    # Количество итераций удаления мелкого шума из расширенной маски.
    expansion_open_iterations: int = 1
    # Количество итераций закрытия разрывов в расширенной маске.
    expansion_close_iterations: int = 2
    # Количество итераций расширения seed-маски для поиска пересечённой компоненты.
    expansion_seed_dilate_iterations: int = 1

    # Верхняя граница площади seed, для которой expansion ratio сильно ограничен.
    expansion_small_area_limit: int = 300
    # Верхняя граница площади seed для среднего ограничения expansion ratio.
    expansion_medium_area_limit: int = 900
    # Верхняя граница площади seed для мягкого ограничения expansion ratio.
    expansion_large_area_limit: int = 1600
    # Максимальный expansion ratio для маленького seed.
    expansion_small_area_ratio: float = 3.0
    # Максимальный expansion ratio для среднего seed.
    expansion_medium_area_ratio: float = 3.2
    # Максимальный expansion ratio для крупного seed.
    expansion_large_area_ratio: float = 4.2

    # Максимальный рост стороны bbox для маленького seed.
    expansion_small_side_ratio: float = 1.8
    # Максимальный рост стороны bbox для среднего seed.
    expansion_medium_side_ratio: float = 1.95
    # Максимальный рост стороны bbox для крупного seed.
    expansion_large_side_ratio: float = 2.3
    # Максимальный рост стороны bbox для остальных seed.
    expansion_default_side_ratio: float = 3.0

    # Aspect ratio, после которого расширение маленькой цели считается подозрительным.
    small_target_expanded_aspect_limit: float = 1.85
    # Множитель ожидаемого размера, после которого ширина расширения считается подозрительной.
    small_target_expanded_width_multiplier: float = 2.35

    def __post_init__(self) -> None:
        """Проверить корректность параметров выбора цели по клику."""
        self._validate_positive_int(self.search_radius, "search_radius")
        self._validate_positive_int(self.max_retry_radius, "max_retry_radius")
        self._validate_positive_float(self.retry_scale, "retry_scale")
        self._validate_non_negative_int(self.padding, "padding")
        self._validate_positive_int(self.fallback_size, "fallback_size")
        self._validate_positive_int(self.min_component_area, "min_component_area")
        self._validate_ratio(self.max_component_fill, "max_component_fill")
        self._validate_ratio(self.max_patch_span_ratio, "max_patch_span_ratio")
        self._validate_positive_float(self.max_refine_growth, "max_refine_growth")
        self._validate_non_negative_float(self.min_object_contrast, "min_object_contrast")

        self._validate_odd_positive_kernel(self.mask_morphology_kernel, "mask_morphology_kernel")
        self._validate_non_negative_int(self.mask_open_iterations, "mask_open_iterations")
        self._validate_non_negative_int(self.mask_close_iterations, "mask_close_iterations")
        self._validate_odd_positive_kernel(self.score_blur_kernel, "score_blur_kernel")
        self._validate_sobel_kernel(self.score_sobel_kernel, "score_sobel_kernel")
        self._validate_non_negative_float(self.score_difference_weight, "score_difference_weight")
        self._validate_non_negative_float(self.score_gradient_weight, "score_gradient_weight")
        self._validate_non_negative_float(self.score_distance_weight, "score_distance_weight")
        self._validate_positive_float(self.score_threshold, "score_threshold")

        self._validate_positive_int(self.snap_min_radius, "snap_min_radius")
        self._validate_positive_float(self.snap_radius_multiplier, "snap_radius_multiplier")
        self._validate_non_negative_int(self.snap_radius_padding, "snap_radius_padding")
        self._validate_odd_positive_kernel(self.snap_blur_kernel, "snap_blur_kernel")
        self._validate_non_negative_float(self.snap_distance_weight, "snap_distance_weight")
        self._validate_positive_float(self.refine_patch_bbox_scale, "refine_patch_bbox_scale")
        self._validate_positive_float(
            self.refine_patch_min_radius_ratio,
            "refine_patch_min_radius_ratio",
        )

        self._validate_non_negative_int(self.background_ring, "background_ring")
        self._validate_positive_int(self.min_background_pixels, "min_background_pixels")
        self._validate_ratio(
            self.compact_selection_confidence,
            "compact_selection_confidence",
        )
        self._validate_positive_int(self.compact_min_radius, "compact_min_radius")
        self._validate_positive_int(self.compact_max_radius, "compact_max_radius")
        self._validate_odd_positive_kernel(self.contrast_blur_kernel, "contrast_blur_kernel")
        self._validate_odd_positive_kernel(
            self.contrast_morphology_kernel,
            "contrast_morphology_kernel",
        )
        self._validate_non_negative_int(self.contrast_open_iterations, "contrast_open_iterations")
        self._validate_non_negative_int(self.contrast_close_iterations, "contrast_close_iterations")

        self._validate_positive_int(self.expansion_margin, "expansion_margin")
        self._validate_positive_float(self.expansion_margin_seed_scale, "expansion_margin_seed_scale")
        self._validate_positive_int(self.expansion_seed_max_area, "expansion_seed_max_area")
        self._validate_ratio(self.max_expanded_fill, "max_expanded_fill")
        self._validate_ratio(self.foreground_fraction, "foreground_fraction")
        self._validate_odd_positive_kernel(
            self.expansion_morphology_kernel,
            "expansion_morphology_kernel",
        )
        self._validate_non_negative_int(self.expansion_open_iterations, "expansion_open_iterations")
        self._validate_non_negative_int(self.expansion_close_iterations, "expansion_close_iterations")
        self._validate_non_negative_int(
            self.expansion_seed_dilate_iterations,
            "expansion_seed_dilate_iterations",
        )

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> Self:
        """Создать конфигурацию из сырых параметров пресета."""
        reader = PresetFieldReader(owner=str(cls.operation_type), values=values)
        kwargs: dict[str, object] = {}

        for field_name in (
            "enabled",
        ):
            reader.pop_bool_to(kwargs, field_name)

        for field_name in (
            "search_radius",
            "max_retry_radius",
            "padding",
            "fallback_size",
            "min_component_area",
            "local_window_radius",
            "min_tolerance",
            "max_tolerance",
            "mask_morphology_kernel",
            "mask_open_iterations",
            "mask_close_iterations",
            "score_blur_kernel",
            "score_sobel_kernel",
            "snap_min_radius",
            "snap_radius_padding",
            "snap_blur_kernel",
            "background_ring",
            "min_background_pixels",
            "compact_min_radius",
            "compact_max_radius",
            "contrast_blur_kernel",
            "contrast_morphology_kernel",
            "contrast_open_iterations",
            "contrast_close_iterations",
            "contrast_split_min_bbox_area",
            "contrast_split_min_bbox_side",
            "contrast_split_min_area_multiplier",
            "contrast_core_min_area",
            "contrast_core_dilate_kernel",
            "contrast_core_dilate_iterations",
            "large_component_min_area_multiplier",
            "tighten_min_area_multiplier",
            "small_target_max_fallback_size",
            "oversized_area_min_multiplier",
            "oversized_area_min_pixels",
            "expansion_margin",
            "expansion_seed_max_area",
            "expansion_morphology_kernel",
            "expansion_open_iterations",
            "expansion_close_iterations",
            "expansion_seed_dilate_iterations",
            "expansion_small_area_limit",
            "expansion_medium_area_limit",
            "expansion_large_area_limit",
        ):
            reader.pop_int_to(kwargs, field_name)

        for field_name in (
            "retry_scale",
            "max_component_fill",
            "max_patch_span_ratio",
            "max_refine_growth",
            "similarity_sigma",
            "min_object_contrast",
            "score_difference_weight",
            "score_gradient_weight",
            "score_distance_weight",
            "score_threshold",
            "snap_radius_multiplier",
            "snap_distance_weight",
            "refine_patch_bbox_scale",
            "refine_patch_min_radius_ratio",
            "compact_selection_confidence",
            "contrast_split_max_patch_fill",
            "contrast_core_min_area_ratio",
            "contrast_split_max_bbox_ratio",
            "split_min_area_ratio",
            "split_improvement_ratio",
            "tighten_min_area_fill",
            "tighten_min_span_ratio",
            "tighten_max_bbox_area_ratio",
            "small_target_max_expansion_ratio",
            "oversized_side_multiplier",
            "oversized_area_multiplier",
            "elongated_aspect_ratio",
            "long_line_search_radius_ratio",
            "max_expansion_ratio",
            "expansion_margin_seed_scale",
            "max_expanded_fill",
            "foreground_fraction",
            "expansion_small_area_ratio",
            "expansion_medium_area_ratio",
            "expansion_large_area_ratio",
            "expansion_small_side_ratio",
            "expansion_medium_side_ratio",
            "expansion_large_side_ratio",
            "expansion_default_side_ratio",
            "small_target_expanded_aspect_limit",
            "small_target_expanded_width_multiplier",
        ):
            reader.pop_float_to(kwargs, field_name)

        reader.pop_float_tuple_to(kwargs, "contrast_split_quantiles")
        reader.pop_float_tuple_to(kwargs, "tighten_tolerance_scales")
        reader.pop_int_tuple_to(kwargs, "split_kernel_sizes")
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
        """Проверить, что значение находится в диапазоне [0, 1]."""
        if not 0 <= value <= 1:
            raise ValueError(f"{field_name} must be in range [0, 1].")

    @staticmethod
    def _validate_odd_positive_kernel(value: int, field_name: str) -> None:
        """Проверить, что размер ядра положительный и нечётный."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd.")

    @staticmethod
    def _validate_sobel_kernel(value: int, field_name: str) -> None:
        """Проверить размер ядра Sobel."""
        if value not in (1, 3, 5, 7):
            raise ValueError(f"{field_name} must be one of: 1, 3, 5, 7.")


@dataclass(slots=True)
class ContrastComponentTargetSelector(BaseTargetSelector):
    """Выбирает цель около точки клика."""

    config: ContrastComponentTargetSelectorConfig
    _patch_extractor: ContrastComponentPatchExtractor = field(init=False, repr=False)
    _mask_builder: ContrastComponentMaskBuilder = field(init=False, repr=False)
    _component_extractor: ContrastComponentExtractor = field(init=False, repr=False)
    _selection_expander: ClickSelectionExpander = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Подготовить внутренние компоненты выбора цели."""
        self._patch_extractor = ContrastComponentPatchExtractor(config=self.config)
        self._mask_builder = ContrastComponentMaskBuilder(config=self.config)
        self._component_extractor = ContrastComponentExtractor(
            config=self.config,
            mask_builder=self._mask_builder,
        )
        self._selection_expander = ClickSelectionExpander(config=self.config)

    def apply(
        self,
        frame: ProcessedFrame,
        point: tuple[int, int],
        expected_bbox: BoundingBox | None = None,
    ) -> TargetSelectorResult:
        """Выбрать цель вокруг точки клика или уточнить уже известную область."""
        attempt_radii = self._build_attempt_radii(expected_bbox)
        fallback: TargetSelectorResult | None = None

        for attempt_index, radius in enumerate(attempt_radii):
            patch = self._patch_extractor.extract_patch(
                image=frame.normalized,
                point=point,
                expected_bbox=expected_bbox,
                radius_override=radius,
            )

            if expected_bbox is None:
                patch = self._patch_extractor.snap_patch_point(patch)
                contrast_selection = self._component_extractor.select_contrast_component(
                    patch=patch,
                    frame_shape=frame.bgr.shape,
                )
                if contrast_selection is not None:
                    return contrast_selection

            mask, polarity = self._mask_builder.build_mask(
                patch=patch.image,
                click_x=patch.local_x,
                click_y=patch.local_y,
            )
            component_bbox, confidence = self._component_extractor.extract_component(
                mask=mask,
                patch=patch,
                expected_bbox=expected_bbox,
            )

            if component_bbox is None and expected_bbox is None:
                score_mask = self._mask_builder.build_score_mask(
                    patch=patch.image,
                    click_x=patch.local_x,
                    click_y=patch.local_y,
                )
                component_bbox, confidence = self._component_extractor.extract_component(
                    mask=score_mask,
                    patch=patch,
                    expected_bbox=expected_bbox,
                )

            if component_bbox is None:
                continue

            selection = TargetSelectorResult(
                bbox=component_bbox.clamp(frame.bgr.shape),
                confidence=confidence,
                polarity=polarity,
            )

            if expected_bbox is None:
                selection = self._selection_expander.expand_selection(
                    patch=patch,
                    selection=selection,
                )

            fallback = selection

            if (
                expected_bbox is None
                and attempt_index + 1 < len(attempt_radii)
                and self._selection_expander.touches_patch_border(selection.bbox, patch)
            ):
                continue

            return selection

        if fallback is not None:
            return fallback

        return self._fallback(
            point=point,
            frame_shape=frame.bgr.shape,
            expected_bbox=expected_bbox,
        )

    def refine(
        self,
        frame: ProcessedFrame,
        bbox: BoundingBox,
    ) -> TargetSelectorResult | None:
        """Аккуратно уточнить уже найденную область цели."""
        selection = self.apply(
            frame=frame,
            point=(int(bbox.center[0]), int(bbox.center[1])),
            expected_bbox=bbox,
        )

        if selection.bbox.area <= 0:
            return None
        if selection.bbox.intersection_over_union(bbox) < 0.15:
            return None
        if selection.bbox.width < max(4, self.config.min_component_area // 2):
            return None
        if selection.bbox.width > int(bbox.width * self.config.max_refine_growth):
            return None
        if selection.bbox.height > int(bbox.height * self.config.max_refine_growth):
            return None
        if selection.bbox.area > int(bbox.area * (self.config.max_refine_growth ** 2)):
            return None

        return selection

    def _build_attempt_radii(
        self,
        expected_bbox: BoundingBox | None,
    ) -> tuple[int, ...]:
        """Подготовить радиусы patch для одной или двух попыток выбора."""
        if expected_bbox is not None:
            radius = max(
                self.config.search_radius // 2,
                int(max(expected_bbox.width, expected_bbox.height) * 0.8)
                + self.config.padding,
            )
            return (radius,)

        base_radius = self.config.search_radius
        retry_radius = min(
            self.config.max_retry_radius,
            int(round(base_radius * self.config.retry_scale)),
        )

        if retry_radius <= base_radius:
            return (base_radius,)

        return base_radius, retry_radius

    def _fallback(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, int] | tuple[int, int, int],
        expected_bbox: BoundingBox | None,
    ) -> TargetSelectorResult:
        """Вернуть запасной bbox, если нормальный выбор цели не сработал."""
        size = self.config.fallback_size

        if expected_bbox is not None:
            size = int(max(expected_bbox.width, expected_bbox.height))

        bbox = BoundingBox.from_center(
            point[0],
            point[1],
            size,
            size,
        ).clamp(frame_shape)

        return TargetSelectorResult(
            bbox=bbox,
            confidence=0.05,
            polarity=TargetPolarity.HOT,
        )
