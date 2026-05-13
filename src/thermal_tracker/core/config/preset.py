"""Загрузка пресетов трекера из TOML-файлов."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib

from ..stages.candidate_filtering.config import CANDIDATE_FILTER_CONFIG_CLASSES, CandidateFilterConfig
from .stage_config import StageConfig
from .stage_config_parser import StageConfigParser

# Имя базового пресета, к которому откатываемся при сомнительном выборе.
DEFAULT_PRESET_NAME = "opencv_general"
# Корень проекта нужен, чтобы искать `presets/` независимо от текущей рабочей папки.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# Здесь лежат все TOML-файлы с пресетами.
PRESETS_DIR = PROJECT_ROOT / "presets"


@dataclass
class PreprocessingConfig:
    """Параметры стадии preprocessing.

    `methods` задаёт последовательность атомарных операций. Параметры ниже
    шарятся между операциями: каждая операция при сборке менеджером забирает
    из этого dataclass только нужные ей поля.
    """

    methods: tuple[str, ...] = (
        "resize",
        "gaussian_blur",
        "median_blur",
        "minmax_normalize",
        "clahe_contrast",
        "gradient",
        "sharpness_metric",
    )
    resize_width: int | None = 960  # Целевая ширина для операции `resize`.
    gaussian_kernel: int = 5  # Размер ядра для `gaussian_blur`.
    median_kernel: int = 3  # Размер ядра для `median_blur`.
    clahe_clip_limit: float = 2.0  # Параметр clipLimit для `clahe_contrast`.
    clahe_tile_grid_size: int = 8  # Размер сетки тайлов для `clahe_contrast`.
    gradient_blur_kernel: int = 3  # Сглаживание перед Sobel в `gradient`.
    bilateral_diameter: int = 7  # Диаметр окрестности для `bilateral`.
    bilateral_sigma_color: float = 40.0  # sigmaColor для `bilateral`.
    bilateral_sigma_space: float = 40.0  # sigmaSpace для `bilateral`.
    percentile_low: float = 2.0  # Нижний перцентиль для `percentile_normalize`.
    percentile_high: float = 98.0  # Верхний перцентиль для `percentile_normalize`.


@dataclass
class GlobalMotionConfig:
    """Параметры грубой оценки движения камеры."""

    enabled: bool = True
    method: str = "opencv_phase_correlation"
    downscale: float = 0.5
    blur_kernel: int = 9
    min_response: float = 0.03
    max_shift_ratio: float = 0.35


@dataclass
class MovingAreaDetectionConfig:
    """Параметры детектора движущихся областей."""

    method: str = "opencv_mog2"


@dataclass
class TargetCandidateExtractionConfig:
    """Параметры сборки кандидатов на цель."""

    method: str = "opencv_connected_components"


@dataclass
class TargetRecoveryConfig:
    """Параметры стадии повторного захвата цели."""

    method: str = "local_template"  # Какой recoverer выбрать через TargetRecovererManager.
    min_lost_frames: int = 5  # После скольких подряд потерянных кадров pipeline дёргает recoverer.
    search_padding: int = 60  # Базовый радиус расширения last_bbox в зону поиска recovery.
    search_padding_growth: int = 8  # Прирост радиуса поиска за каждый дополнительный потерянный кадр.
    scales: tuple[float, ...] = (0.84, 0.94, 1.0, 1.08, 1.18)  # Масштабы шаблона при поиске.
    match_threshold: float = 0.5  # Минимальный score (TM_CCOEFF_NORMED), при котором считаем кандидата найденным.
    template_alpha: float = 0.12  # Скорость обновления адаптивного шаблона при remember().
    confirm_frames: int = 3  # Сколько подряд согласованных кадров нужно для перехода RECOVERING -> TRACKING.
    recovery_window_frames: int = 30  # Максимальная длина окна RECOVERING; после него pipeline уходит в LOST.
    max_speed_px_per_frame: float = 0.0  # Физический предел скорости объекта при recovery (px/кадр).
    # Значение 0.0 = gate отключён (backward-compatible для opencv-пресетов).
    # Для IRST-пресетов задаётся явно через [target_recovery] в TOML.


@dataclass
class ClickSelectionConfig:
    """Параметры начального выделения цели по одному клику."""

    method: str = "opencv_click"
    search_radius: int = 80
    similarity_sigma: float = 1.35
    local_window_radius: int = 5
    min_tolerance: int = 10
    max_tolerance: int = 42
    min_component_area: int = 30
    max_component_fill: float = 0.45
    max_patch_span_ratio: float = 0.85
    max_refine_growth: float = 1.8
    retry_scale: float = 1.6
    max_retry_radius: int = 180
    padding: int = 8
    fallback_size: int = 36
    expansion_margin: int = 18
    background_ring: int = 10
    foreground_fraction: float = 0.36
    min_object_contrast: float = 6.0
    max_expanded_fill: float = 0.58
    max_expansion_ratio: float = 5.5


@dataclass
class OpenCVTrackerConfig:
    """Параметры classical single-target трекера (opencv_template_point).

    Все поля используются исключительно ClickToTrackSingleTargetTracker.
    Секция TOML: [opencv_tracking].
    """

    search_margin: int = 24
    lost_search_growth: int = 18
    # Намеренно большое значение: явный full-frame switch отключён.
    # Expanding margin (search_margin + lost_frames * lost_search_growth) покрывает весь кадр
    # органически до истечения max_lost_frames, не создавая резкого скачка зоны поиска.
    full_frame_after: int = 999
    max_lost_frames: int = 70
    scales: tuple[float, ...] = (0.72, 0.84, 0.94, 1.0, 1.08, 1.18, 1.32)
    track_threshold: float = 0.42
    reacquire_threshold: float = 0.5
    template_update_threshold: float = 0.62
    template_alpha: float = 0.12
    velocity_alpha: float = 0.45
    min_box_size: int = 8
    distance_penalty: float = 0.14
    max_size_growth: float = 1.25
    max_size_shrink: float = 0.72
    max_size_growth_on_reacquire: float = 1.6
    max_size_shrink_on_reacquire: float = 0.55
    max_size_growth_from_initial: float = 4.0
    max_feature_points: int = 40
    min_feature_points: int = 6
    feature_quality_level: float = 0.02
    feature_min_distance: int = 6
    feature_refresh_interval: int = 6
    point_search_margin: int = 14
    max_tracking_center_shift: float = 1.6
    max_reacquire_center_shift: float = 2.8
    reacquire_center_growth: float = 0.32
    edge_exit_margin: int = 10
    edge_exit_max_lost_frames: int = 4
    blur_hold_enabled: bool = True
    blur_sharpness_drop_ratio: float = 0.5
    blur_hold_max_frames: int = 120
    blur_hold_center_growth: float = 0.06


@dataclass
class YoloTrackerConfig:
    """Параметры NN single-target трекера (nn_yolo).

    Содержит только поля, которые реально читает YoloTrackSingleTargetTracker.
    Секция TOML: [yolo_tracking].
    """

    max_lost_frames: int = 90
    search_margin: int = 30
    lost_search_growth: int = 26


@dataclass(frozen=True)
class IrstTrackerConfig:
    """Параметры IRST-трекера (irst_contrast) на базе локального контраста и фильтра Калмана.

    IRST — Infrared Search and Track / инфракрасный поиск и сопровождение.
    Вместо сопоставления шаблонов ищет пиксельный кластер, который значительно
    ярче или темнее своего локального фона. Позиция измеряется центроидом кластера.
    Движение предсказывается фильтром Калмана с моделью постоянной скорости.

    Секция TOML: [irst_tracking].
    """

    # --- Детектор локального контраста ---
    filter_kernel: int = 3
    """Размер внутреннего окна для поиска локального максимума/минимума (пиксели)."""
    background_kernel: int = 11
    """Размер внешнего окна для оценки локального фона (пиксели, нечётное число)."""
    contrast_threshold: float = 12.0
    """Минимальный контраст (в единицах 0–255), при котором пиксель считается частью цели."""
    min_blob_area: int = 1
    """Минимальная площадь кластера-кандидата в пикселях."""
    max_blob_area: int = 200
    """Максимальная площадь кластера-кандидата в пикселях."""

    # --- Фильтр Калмана (модель постоянной скорости: cx, cy, vx, vy) ---
    kalman_process_noise_pos: float = 0.5
    """Шум процесса по позиции (насколько точно модель описывает движение)."""
    kalman_process_noise_vel: float = 2.0
    """Шум процесса по скорости (насколько допускаем ускорение цели)."""
    kalman_measurement_noise: float = 1.5
    """Шум измерения (ошибка определения центроида blob в пикселях)."""

    # --- Зона захвата (gate) ---
    min_gate: int = 25
    """Минимальный радиус поиска кандидата вокруг прогноза Калмана (пиксели)."""
    gate_growth: int = 4
    """Расширение радиуса поиска за каждый пропущенный кадр (пиксели/кадр)."""
    max_gate: int = 120
    """Максимальный радиус поиска (физический предел, пиксели)."""
    max_speed_px_per_frame: float = 8.0
    """Жёсткий физический предел скорости объекта (пиксели/кадр).
    Кандидат за пределами last_good_bbox + max_speed * lost_frames отклоняется."""

    # --- Устойчивость к потере ---
    max_lost_frames: int = 30
    """Максимальный пропуск кадров без обнаружения, после которого трекер уходит в IDLE."""

    # --- Обработка замутнения ---
    blur_hold_enabled: bool = True
    """Включить режим удержания прогноза при размытом кадре."""
    blur_sharpness_drop_ratio: float = 0.60
    """Если резкость кадра упала ниже baseline * ratio, считаем кадр замутнённым."""

    # --- Выбор цели кликом ---
    click_search_radius: int = 60
    """Радиус поиска ближайшего blob к точке клика (пиксели)."""
    click_fallback_size: int = 12
    """Размер запасного бокса, если рядом с кликом нет кандидатов."""


@dataclass
class VisualizationConfig:
    """Параметры отрисовки поверх кадра."""

    show_search_region: bool = True
    show_predicted_box: bool = True
    show_global_motion: bool = True
    line_thickness: int = 1


@dataclass
class NeuralConfig:
    """Параметры нейросетевого контура."""

    engine_name: str = "ultralytics_yolo"
    model_path: str = ""
    tracker_config_path: str = ""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    allowed_classes: tuple[int, ...] = ()
    device: str = ""
    max_reacquire_distance_factor: float = 2.6
    prefer_same_class: bool = True


@dataclass
class TrackerPreset:
    """Готовый набор настроек для конкретного сценария."""

    name: str
    preprocessing: PreprocessingConfig
    global_motion: GlobalMotionConfig
    moving_area_detection: MovingAreaDetectionConfig
    target_candidate_extraction: TargetCandidateExtractionConfig
    target_recovery: TargetRecoveryConfig
    candidate_filtering: StageConfig[CandidateFilterConfig]
    click_selection: ClickSelectionConfig
    visualization: VisualizationConfig
    pipeline_kind: str = "manual_click_classical"
    opencv_tracker: OpenCVTrackerConfig | None = None
    yolo_tracker: YoloTrackerConfig | None = None
    irst_tracker: IrstTrackerConfig | None = None
    neural: NeuralConfig | None = None


@dataclass(frozen=True)
class PresetPresentation:
    """Человеческое описание пресета для GUI."""

    title: str
    tooltip: str
    description: str


@dataclass(frozen=True)
class _PresetRecord:
    """Полный набор данных одного загруженного пресета."""

    preset: TrackerPreset
    presentation: PresetPresentation
    file_path: Path


def _read_toml(path: Path) -> dict[str, object]:
    """Читает TOML-файл пресета и возвращает его как словарь."""

    with path.open("rb") as file:
        loaded = tomllib.load(file)

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Пресет {path} не удалось прочитать как TOML-словарь.")
    return loaded


def _normalize_opencv_tracker_section(section: dict[str, object]) -> dict[str, object]:
    """Подготавливает значения секции `[opencv_tracking]` перед созданием OpenCVTrackerConfig."""

    normalized = dict(section)
    scales = normalized.get("scales")
    if scales is not None:
        normalized["scales"] = tuple(float(value) for value in scales)
    return normalized


def _normalize_neural_section(section: dict[str, object]) -> dict[str, object]:
    """Подготавливает значения секции `neural` перед созданием dataclass."""

    normalized = dict(section)
    allowed_classes = normalized.get("allowed_classes")
    if allowed_classes is not None:
        normalized["allowed_classes"] = tuple(int(value) for value in allowed_classes)
    return normalized


def _normalize_preprocessing_section(section: dict[str, object]) -> dict[str, object]:
    """Подготавливает список атомарных операций preprocessing перед созданием dataclass."""

    normalized = dict(section)
    methods = normalized.get("methods")
    if methods is not None:
        if isinstance(methods, str):
            normalized["methods"] = (methods,)
        else:
            normalized["methods"] = tuple(str(value) for value in methods)
    return normalized


def _normalize_target_recovery_section(section: dict[str, object]) -> dict[str, object]:
    """Подготавливает значения секции `target_recovery` перед созданием dataclass."""

    normalized = dict(section)
    scales = normalized.get("scales")
    if scales is not None:
        normalized["scales"] = tuple(float(value) for value in scales)
    return normalized


def _build_presentation(preset_name: str, meta: dict[str, object]) -> PresetPresentation:
    """Собирает описание пресета для интерфейса."""

    title = str(meta.get("title") or preset_name)
    tooltip = str(meta.get("tooltip") or f"Пресет {preset_name}")
    description = str(meta.get("description") or tooltip)
    return PresetPresentation(title=title, tooltip=tooltip, description=description)


def _build_preset_record(path: Path) -> _PresetRecord:
    """Создаёт полное описание пресета из одного TOML-файла."""

    data = _read_toml(path)
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise RuntimeError(f"В пресете {path} секция [meta] должна быть таблицей.")

    pipeline = data.get("pipeline", {})
    if pipeline is None:
        pipeline = {}
    if not isinstance(pipeline, dict):
        raise RuntimeError(f"В пресете {path} секция [pipeline] должна быть таблицей.")

    preset_name = str(meta.get("id") or path.stem).strip()
    if not preset_name:
        raise RuntimeError(f"В пресете {path} не задано имя.")

    preprocessing = PreprocessingConfig(
        **_normalize_preprocessing_section(dict(data.get("preprocessing", {})))
    )
    global_motion = GlobalMotionConfig(**dict(data.get("global_motion", {})))
    moving_area_detection = MovingAreaDetectionConfig(**dict(data.get("moving_area_detection", {})))
    target_candidate_extraction = TargetCandidateExtractionConfig(
        **dict(data.get("target_candidate_extraction", {}))
    )
    target_recovery = TargetRecoveryConfig(
        **_normalize_target_recovery_section(dict(data.get("target_recovery", {})))
    )

    candidate_filtering = StageConfigParser.parse(
        section=dict(data.get("candidate_filtering", {})),
        stage_name="candidate_filtering",
        config_classes=CANDIDATE_FILTER_CONFIG_CLASSES,
    )

    click_selection = ClickSelectionConfig(**dict(data.get("click_selection", {})))
    visualization = VisualizationConfig(**dict(data.get("visualization", {})))
    pipeline_kind = str(pipeline.get("kind") or "manual_click_classical").strip() or "manual_click_classical"

    opencv_tracking_data = data.get("opencv_tracking")
    opencv_tracker: OpenCVTrackerConfig | None = None
    if isinstance(opencv_tracking_data, dict):
        opencv_tracker = OpenCVTrackerConfig(
            **_normalize_opencv_tracker_section(opencv_tracking_data)
        )

    yolo_tracking_data = data.get("yolo_tracking")
    yolo_tracker: YoloTrackerConfig | None = None
    if isinstance(yolo_tracking_data, dict):
        yolo_tracker = YoloTrackerConfig(**dict(yolo_tracking_data))

    irst_tracking_data = data.get("irst_tracking")
    irst_tracker: IrstTrackerConfig | None = None
    if isinstance(irst_tracking_data, dict):
        irst_tracker = IrstTrackerConfig(**dict(irst_tracking_data))

    neural_section = data.get("neural")
    neural = None
    if isinstance(neural_section, dict) and neural_section:
        neural = NeuralConfig(**_normalize_neural_section(neural_section))

    return _PresetRecord(
        preset=TrackerPreset(
            name=preset_name,
            preprocessing=preprocessing,
            global_motion=global_motion,
            moving_area_detection=moving_area_detection,
            target_candidate_extraction=target_candidate_extraction,
            target_recovery=target_recovery,
            candidate_filtering=candidate_filtering,
            click_selection=click_selection,
            visualization=visualization,
            pipeline_kind=pipeline_kind,
            opencv_tracker=opencv_tracker,
            yolo_tracker=yolo_tracker,
            irst_tracker=irst_tracker,
            neural=neural,
        ),
        presentation=_build_presentation(preset_name, meta),
        file_path=path,
    )


@lru_cache(maxsize=1)
def _load_preset_catalog() -> dict[str, _PresetRecord]:
    """Читает все TOML-пресеты из корневой папки `presets`."""

    if not PRESETS_DIR.exists():
        raise RuntimeError(f"Каталог с пресетами не найден: {PRESETS_DIR}")

    catalog: dict[str, _PresetRecord] = {}
    for path in sorted(PRESETS_DIR.glob("*.toml")):
        data = _read_toml(path)
        if "meta" not in data:
            continue
        record = _build_preset_record(path)
        if record.preset.name in catalog:
            raise RuntimeError(f"Имя пресета {record.preset.name!r} повторяется в {path.name}.")
        catalog[record.preset.name] = record

    if not catalog:
        raise RuntimeError(f"В каталоге {PRESETS_DIR} не найдено ни одного TOML-пресета.")
    return catalog


def _resolve_preset_name(name: str) -> str:
    """Приводит имя пресета к реально существующему варианту."""

    requested = (name or "").strip()
    catalog = _load_preset_catalog()
    if requested in catalog:
        return requested
    if DEFAULT_PRESET_NAME in catalog:
        return DEFAULT_PRESET_NAME
    return next(iter(catalog))


def get_available_preset_names() -> tuple[str, ...]:
    """Возвращает список доступных пресетов в порядке загрузки."""

    catalog = _load_preset_catalog()
    ordered_names: list[str] = []

    if DEFAULT_PRESET_NAME in catalog:
        ordered_names.append(DEFAULT_PRESET_NAME)

    ordered_names.extend(name for name in catalog.keys() if name != DEFAULT_PRESET_NAME)
    return tuple(ordered_names)


def build_preset(name: str) -> TrackerPreset:
    """Возвращает полную конфигурацию пресета по имени."""

    preset_name = _resolve_preset_name(name)
    return deepcopy(_load_preset_catalog()[preset_name].preset)


def get_preset_presentation(name: str) -> PresetPresentation:
    """Возвращает описание пресета для интерфейса."""

    preset_name = _resolve_preset_name(name)
    return _load_preset_catalog()[preset_name].presentation


# Готовый список имён для GUI и CLI, чтобы не грузить каталог вручную в каждом месте.
AVAILABLE_PRESETS = get_available_preset_names()
