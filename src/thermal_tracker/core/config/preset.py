"""Загрузка пресетов трекера из TOML-файлов."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib

# Имя базового пресета, к которому откатываемся при сомнительном выборе.
DEFAULT_PRESET_NAME = "opencv_general"
# Корень проекта нужен, чтобы искать `presets/` независимо от текущей рабочей папки.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Здесь лежат все TOML-файлы с пресетами.
PRESETS_DIR = PROJECT_ROOT / "presets"


@dataclass
class PreprocessingConfig:
    """Параметры базовой подготовки кадра."""
    resize_width: int | None = 960  # Если кадр слишком широкий, уменьшаем его до этой ширины.
    gaussian_kernel: int = 5  # Размер ядра мягкого гауссова сглаживания.
    median_kernel: int = 3  # Размер ядра медианного фильтра против одиночного шума.
    clahe_clip_limit: float = 2.0  # Насколько агрессивно усиливаем локальный контраст.
    clahe_tile_grid_size: int = 8  # На сколько плиток делим кадр для CLAHE.
    gradient_blur_kernel: int = 3  # Дополнительное сглаживание перед расчётом градиентов.


@dataclass
class GlobalMotionConfig:
    """Параметры грубой оценки движения камеры."""
    enabled: bool = True  # Включать ли оценку глобального сдвига камеры.
    downscale: float = 0.5  # Во сколько раз уменьшать кадр перед фазовой корреляцией.
    blur_kernel: int = 9  # Размер сглаживания перед оценкой общего движения.
    min_response: float = 0.03  # Минимальная надёжность ответа phase correlation.
    max_shift_ratio: float = 0.35  # Максимально допустимый сдвиг как доля размера кадра.


@dataclass
class ClickSelectionConfig:
    """Параметры начального выделения цели по одному клику."""
    search_radius: int = 80  # Базовый радиус поиска объекта вокруг точки клика.
    similarity_sigma: float = 1.35  # Насколько широко разрешаем разброс яркости вокруг клика.
    local_window_radius: int = 5  # Размер локального окна для оценки местной статистики.
    min_tolerance: int = 10  # Нижняя граница допуска по яркости.
    max_tolerance: int = 42  # Верхняя граница допуска по яркости.
    min_component_area: int = 30  # Минимальная площадь компоненты, которую считаем объектом.
    max_component_fill: float = 0.45  # Какую долю патча может занимать одна компонента, прежде чем станет подозрительной.
    max_patch_span_ratio: float = 0.85  # Насколько компоненте можно расползтись по ширине или высоте патча.
    max_refine_growth: float = 1.8  # Во сколько раз уточнение может расширить уже известный bbox.
    retry_scale: float = 1.6  # Во сколько раз увеличиваем радиус, если первая попытка не удалась.
    max_retry_radius: int = 180  # Максимальный радиус повторной попытки выбора цели.
    padding: int = 8  # Сколько пикселей добавляем вокруг найденной компоненты.
    fallback_size: int = 36  # Размер запасного бокса, если сегментация не удалась.
    expansion_margin: int = 18  # Запас вокруг ядра цели при попытке расширить его до целого объекта.
    background_ring: int = 10  # Толщина внешнего кольца фона для сравнения контраста.
    foreground_fraction: float = 0.36  # Доля контраста, по которой строим порог объекта относительно фона.
    min_object_contrast: float = 6.0  # Минимальный контраст объекта к фону для уверенного расширения.
    max_expanded_fill: float = 0.58  # Максимальная доля рабочей области, которую может занять расширенный объект.
    max_expansion_ratio: float = 5.5  # Во сколько раз расширенный объект может быть больше начального ядра.


@dataclass
class TrackerConfig:
    """Параметры самого трекера и повторного поиска цели."""
    search_margin: int = 30  # Начальный запас вокруг предсказанной позиции при обычном поиске.
    lost_search_growth: int = 26  # На сколько расширяем зону поиска при потере цели.
    full_frame_after: int = 14  # После скольких пропусков разрешаем почти полный поиск по кадру.
    max_lost_frames: int = 90  # Сколько кадров можно терпеть потерю цели до окончательного провала.
    scales: tuple[float, ...] = (0.82, 0.94, 1.0, 1.08, 1.18)  # Набор масштабов бокса для перебора шаблона.
    track_threshold: float = 0.42  # Минимальный порог совпадения для обычного сопровождения.
    reacquire_threshold: float = 0.5  # Минимальный порог совпадения для повторного захвата.
    template_update_threshold: float = 0.62  # Порог уверенности, после которого можно обновлять шаблон.
    template_alpha: float = 0.12  # Скорость обновления адаптивного шаблона цели.
    velocity_alpha: float = 0.45  # Насколько сильно учитываем новую оценку скорости цели.
    min_box_size: int = 12  # Самый маленький допустимый размер бокса.
    distance_penalty: float = 0.18  # Штраф за кандидатов, которые далеко от предсказанной позиции.
    max_size_growth: float = 1.25  # Максимальный рост бокса за один обычный шаг.
    max_size_shrink: float = 0.72  # Максимальное сжатие бокса за один обычный шаг.
    max_size_growth_on_reacquire: float = 1.6  # Насколько можно увеличить бокс при повторном захвате.
    max_size_shrink_on_reacquire: float = 0.55  # Насколько можно уменьшить бокс при повторном захвате.
    max_size_growth_from_initial: float = 4.0  # Во сколько раз бокс может вырасти относительно стартового размера.
    max_feature_points: int = 40  # Верхняя граница числа опорных точек для LK-трекинга.
    min_feature_points: int = 6  # Нижняя граница числа опорных точек, ниже которой пора переинициализироваться.
    feature_quality_level: float = 0.02  # Порог качества углов при поиске опорных точек.
    feature_min_distance: int = 6  # Минимальная дистанция между опорными точками.
    feature_refresh_interval: int = 6  # Как часто полностью обновлять набор опорных точек.
    point_search_margin: int = 14  # Запас вокруг бокса для поиска новых точек.
    max_tracking_center_shift: float = 1.45  # Максимальный допустимый сдвиг центра в обычном режиме.
    max_reacquire_center_shift: float = 2.4  # Максимальный допустимый сдвиг центра при повторном захвате.
    reacquire_center_growth: float = 0.28  # Как быстро расширяем допуск по центру при затянувшейся потере.
    edge_exit_margin: int = 10  # Насколько близко к краю кадра цель считаем уходящей за границу.
    edge_exit_max_lost_frames: int = 4  # Сколько кадров искать цель после потери прямо у края кадра.
    blur_hold_enabled: bool = True  # Включать ли режим удержания прогноза при заметном замутнении кадра.
    blur_sharpness_drop_ratio: float = 0.55  # Насколько должна просесть резкость относительно нормального кадра.
    blur_hold_max_frames: int = 70  # Сколько дополнительных кадров можно держать прогноз после замутнения.
    blur_hold_center_growth: float = 0.07  # Как расширять допуск по центру во время восстановления после замутнения.


@dataclass
class VisualizationConfig:
    """Параметры отрисовки поверх кадра."""
    show_search_region: bool = True  # Показывать ли зону, в которой трекер сейчас ищет цель.
    show_predicted_box: bool = True  # Показывать ли предсказанный бокс до финального совпадения.
    show_global_motion: bool = True  # Держать ли в статусе информацию о движении камеры.
    line_thickness: int = 1  # Толщина линий при рисовании боксов.


@dataclass
class NeuralConfig:
    """Параметры нейросетевого контура."""
    engine_name: str = "ultralytics_yolo"  # Какой backend инференса использовать.
    model_path: str = ""  # Путь к файлу весов модели относительно корня проекта или абсолютный.
    tracker_config_path: str = ""  # Путь к YAML-конфигу внешнего трекера, например ByteTrack.
    confidence_threshold: float = 0.25  # Минимальная уверенность детекции, ниже которой её игнорируем.
    iou_threshold: float = 0.45  # Порог IoU для внутренней NMS модели, если backend его поддерживает.
    allowed_classes: tuple[int, ...] = ()  # Список разрешённых классов. Пусто значит брать всё.
    device: str = ""  # Устройство инференса. Пусто значит пусть backend выберет сам.
    max_reacquire_distance_factor: float = 2.6  # Насколько далеко от прошлого бокса можно искать ту же цель.
    prefer_same_class: bool = True  # Стараться ли при повторном захвате держаться того же класса.


@dataclass
class TrackerPreset:
    """Готовый набор настроек для конкретного сценария."""
    name: str  # Короткое техническое имя пресета.
    preprocessing: PreprocessingConfig  # Настройки предобработки кадра.
    global_motion: GlobalMotionConfig  # Настройки оценки движения камеры.
    click_selection: ClickSelectionConfig  # Настройки старта цели по клику.
    tracker: TrackerConfig  # Основные настройки сопровождения и повторного захвата.
    visualization: VisualizationConfig  # Настройки отрисовки служебной информации.
    pipeline_kind: str = "manual_click_classical"  # Какой pipeline собирать для этого пресета.
    neural: NeuralConfig | None = None  # Параметры нейросетевого контура, если он нужен.


@dataclass(frozen=True)
class PresetPresentation:
    """Человеческое описание пресета для GUI."""
    title: str  # Короткое имя, которое можно показать рядом со списком пресетов.
    tooltip: str  # Короткая подсказка для hover и компактных мест GUI.
    description: str  # Нормальное человеческое описание того, когда использовать пресет.


@dataclass(frozen=True)
class _PresetRecord:
    """Полный набор данных одного загруженного пресета."""
    preset: TrackerPreset  # Полная техническая конфигурация пресета.
    presentation: PresetPresentation  # Человеческое описание для интерфейса.
    file_path: Path  # Откуда именно был прочитан этот пресет.


def _read_toml(path: Path) -> dict[str, object]:
    """Читает TOML-файл пресета и возвращает его как словарь."""
    with path.open("rb") as file:
        loaded = tomllib.load(file)

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Пресет {path} не удалось прочитать как TOML-словарь.")
    return loaded


def _normalize_tracker_section(section: dict[str, object]) -> dict[str, object]:
    """Подготавливает значения секции `tracking` перед созданием dataclass."""
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

    preprocessing = PreprocessingConfig(**dict(data.get("preprocessing", {})))
    global_motion = GlobalMotionConfig(**dict(data.get("global_motion", {})))
    click_selection = ClickSelectionConfig(**dict(data.get("click_selection", {})))
    tracker = TrackerConfig(**_normalize_tracker_section(dict(data.get("tracking", {}))))
    visualization = VisualizationConfig(**dict(data.get("visualization", {})))
    pipeline_kind = str(pipeline.get("kind") or "manual_click_classical").strip() or "manual_click_classical"

    neural_section = data.get("neural")
    neural = None
    if isinstance(neural_section, dict) and neural_section:
        neural = NeuralConfig(**_normalize_neural_section(neural_section))

    return _PresetRecord(
        preset=TrackerPreset(
            name=preset_name,
            preprocessing=preprocessing,
            global_motion=global_motion,
            click_selection=click_selection,
            tracker=tracker,
            visualization=visualization,
            pipeline_kind=pipeline_kind,
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
