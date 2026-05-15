"""Загрузка пресетов трекера из TOML-файлов."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
import tomllib

from thermal_tracker.core.stages.config.stage_config import StageConfig
from thermal_tracker.core.stages.config.stage_config_parser import StageConfigParser

if TYPE_CHECKING:
    from ..stages.candidate_filtering import CandidateFilterConfig
    from ..stages.candidate_formation import CandidateFormerConfig
    from ..stages.frame_preprocessing import FramePreprocessorConfig
    from ..stages.frame_stabilization import FrameStabilizerConfig
    from ..stages.motion_localization import MotionLocalizationConfig
    from ..stages.target_recovery import TargetRecovererConfig
    from ..stages.target_selection import TargetSelectionConfig
    from ..stages.target_tracking import TargetTrackerConfig


# Имя базового пресета, к которому откатываемся при сомнительном выборе.
DEFAULT_PRESET_NAME = "opencv_general"
# Корень проекта нужен, чтобы искать `presets/` независимо от текущей рабочей папки.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# Здесь лежат все TOML-файлы с пресетами.
PRESETS_DIR = PROJECT_ROOT / "presets"


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Параметры отрисовки поверх кадра."""

    show_search_region: bool = True
    show_predicted_box: bool = True
    show_global_motion: bool = True
    line_thickness: int = 1

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> VisualizationConfig:
        """Создать конфигурацию отрисовки из TOML-секции."""
        kwargs: dict[str, object] = {}
        source = dict(values)

        for field_name in (
            "show_search_region",
            "show_predicted_box",
            "show_global_motion",
        ):
            value = source.pop(field_name, None)
            if value is None:
                continue
            if not isinstance(value, bool):
                raise RuntimeError(f"visualization.{field_name} must be boolean.")
            kwargs[field_name] = value

        line_thickness = source.pop("line_thickness", None)
        if line_thickness is not None:
            if isinstance(line_thickness, bool) or not isinstance(line_thickness, int):
                raise RuntimeError("visualization.line_thickness must be integer.")
            kwargs["line_thickness"] = line_thickness

        if source:
            raise RuntimeError(f"Unsupported visualization params: {tuple(sorted(source))}.")

        return cls(**kwargs)

    def __post_init__(self) -> None:
        """Проверить параметры отрисовки."""
        if self.line_thickness <= 0:
            raise ValueError("line_thickness must be greater than 0.")


@dataclass(frozen=True, slots=True)
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

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> NeuralConfig:
        """Создать конфигурацию нейросетевого контура из TOML-секции."""
        source = dict(values)
        kwargs: dict[str, object] = {}

        for field_name in (
            "engine_name",
            "model_path",
            "tracker_config_path",
            "device",
        ):
            value = source.pop(field_name, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise RuntimeError(f"neural.{field_name} must be string.")
            kwargs[field_name] = value

        for field_name in (
            "confidence_threshold",
            "iou_threshold",
            "max_reacquire_distance_factor",
        ):
            value = source.pop(field_name, None)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"neural.{field_name} must be number.")
            kwargs[field_name] = float(value)

        prefer_same_class = source.pop("prefer_same_class", None)
        if prefer_same_class is not None:
            if not isinstance(prefer_same_class, bool):
                raise RuntimeError("neural.prefer_same_class must be boolean.")
            kwargs["prefer_same_class"] = prefer_same_class

        allowed_classes = source.pop("allowed_classes", None)
        if allowed_classes is not None:
            if not isinstance(allowed_classes, (list, tuple)):
                raise RuntimeError("neural.allowed_classes must be array.")
            parsed_classes: list[int] = []
            for item in allowed_classes:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise RuntimeError("neural.allowed_classes items must be integer.")
                parsed_classes.append(item)
            kwargs["allowed_classes"] = tuple(parsed_classes)

        if source:
            raise RuntimeError(f"Unsupported neural params: {tuple(sorted(source))}.")

        return cls(**kwargs)

    def __post_init__(self) -> None:
        """Проверить параметры нейросетевого контура."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in range [0, 1].")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in range [0, 1].")
        if self.max_reacquire_distance_factor <= 0.0:
            raise ValueError("max_reacquire_distance_factor must be greater than 0.")


@dataclass(frozen=True, slots=True)
class TargetRecoveryConfig:
    """Конфигурация стадии повторного захвата цели."""

    # Включает или отключает всю стадию восстановления цели.
    enabled: bool = False
    # Операции восстановления цели в порядке fallback-цепочки.
    operations: tuple[TargetRecovererConfig, ...] = ()
    # После скольких потерянных кадров pipeline начинает recovery.
    min_lost_frames: int = 5
    # Сколько подряд подтверждений нужно для перехода RECOVERING -> TRACKING.
    confirm_frames: int = 3
    # Максимальная длина окна восстановления перед окончательной потерей цели.
    recovery_window_frames: int = 30

    @property
    def enabled_operations(self) -> tuple[TargetRecovererConfig, ...]:
        """Вернуть активные операции recovery."""
        if not self.enabled:
            return ()
        return self.operations

    def __post_init__(self) -> None:
        """Проверить параметры стадии восстановления цели."""
        if self.enabled and not self.operations:
            raise ValueError("Target recovery is enabled, but no operations are configured.")
        if self.min_lost_frames < 0:
            raise ValueError("min_lost_frames must be greater than or equal to 0.")
        if self.confirm_frames <= 0:
            raise ValueError("confirm_frames must be greater than 0.")
        if self.recovery_window_frames <= 0:
            raise ValueError("recovery_window_frames must be greater than 0.")


@dataclass(frozen=True, slots=True)
class TrackerPreset:
    """Готовый набор настроек для конкретного сценария."""

    name: str
    frame_preprocessing: StageConfig[FramePreprocessorConfig]
    frame_stabilization: StageConfig[FrameStabilizerConfig]
    motion_localization: StageConfig[MotionLocalizationConfig]
    candidate_formation: StageConfig[CandidateFormerConfig]
    candidate_filtering: StageConfig[CandidateFilterConfig]
    target_selection: StageConfig[TargetSelectionConfig]
    target_tracking: StageConfig[TargetTrackerConfig]
    target_recovery: TargetRecoveryConfig
    visualization: VisualizationConfig
    pipeline_kind: str = "manual_click_classical"
    neural: NeuralConfig | None = None

    @property
    def preprocessing(self) -> StageConfig[FramePreprocessorConfig]:
        """Совместимый алиас старого имени preprocessing."""
        return self.frame_preprocessing

    @property
    def global_motion(self) -> StageConfig[FrameStabilizerConfig]:
        """Совместимый алиас старого имени global_motion."""
        return self.frame_stabilization

    @property
    def moving_area_detection(self) -> StageConfig[MotionLocalizationConfig]:
        """Совместимый алиас старого имени moving_area_detection."""
        return self.motion_localization

    @property
    def target_candidate_extraction(self) -> StageConfig[CandidateFormerConfig]:
        """Совместимый алиас старого имени target_candidate_extraction."""
        return self.candidate_formation

    @property
    def click_selection(self) -> StageConfig[TargetSelectionConfig]:
        """Совместимый алиас старого имени click_selection."""
        return self.target_selection


@dataclass(frozen=True, slots=True)
class PresetPresentation:
    """Человеческое описание пресета для GUI."""

    title: str
    tooltip: str
    description: str


@dataclass(frozen=True, slots=True)
class _PresetIndexRecord:
    """Краткие данные пресета без полной сборки stage-конфигов."""

    name: str
    presentation: PresetPresentation
    file_path: Path


@dataclass(frozen=True, slots=True)
class _PresetRecord:
    """Полный набор данных одного загруженного пресета."""

    preset: TrackerPreset
    presentation: PresetPresentation
    file_path: Path


def _read_toml(path: Path) -> dict[str, object]:
    """Прочитать TOML-файл пресета и вернуть его как словарь."""
    with path.open("rb") as file:
        loaded = tomllib.load(file)

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Preset {path} was not loaded as a TOML dictionary.")

    return loaded


def _build_presentation(preset_name: str, meta: dict[str, object]) -> PresetPresentation:
    """Собрать описание пресета для интерфейса."""
    title = str(meta.get("title") or preset_name)
    tooltip = str(meta.get("tooltip") or f"Preset {preset_name}")
    description = str(meta.get("description") or tooltip)

    return PresetPresentation(title=title, tooltip=tooltip, description=description)


def _parse_meta(path: Path, data: dict[str, object]) -> tuple[str, PresetPresentation]:
    """Прочитать идентификатор и описание пресета."""
    meta = data.get("meta", {})

    if not isinstance(meta, dict):
        raise RuntimeError(f"Preset {path} section [meta] must be a TOML table.")

    preset_name = str(meta.get("id") or path.stem).strip()
    if not preset_name:
        raise RuntimeError(f"Preset {path} has empty id.")

    return preset_name, _build_presentation(preset_name=preset_name, meta=meta)


def _parse_pipeline_kind(data: dict[str, object], path: Path) -> str:
    """Прочитать тип pipeline из секции [pipeline]."""
    pipeline = data.get("pipeline", {})

    if pipeline is None:
        return "manual_click_classical"

    if not isinstance(pipeline, dict):
        raise RuntimeError(f"Preset {path} section [pipeline] must be a TOML table.")

    return str(pipeline.get("kind") or "manual_click_classical").strip() or "manual_click_classical"


def _parse_frame_preprocessing(data: dict[str, object]) -> StageConfig[FramePreprocessorConfig]:
    """Прочитать конфигурацию стадии frame_preprocessing."""
    from ..stages.frame_preprocessing import FRAME_PREPROCESSOR_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("frame_preprocessing", {})),
        stage_name="frame_preprocessing",
        config_classes=FRAME_PREPROCESSOR_CONFIG_CLASSES,
    )


def _parse_frame_stabilization(data: dict[str, object]) -> StageConfig[FrameStabilizerConfig]:
    """Прочитать конфигурацию стадии frame_stabilization."""
    from ..stages.frame_stabilization import FRAME_STABILIZER_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("frame_stabilization", {})),
        stage_name="frame_stabilization",
        config_classes=FRAME_STABILIZER_CONFIG_CLASSES,
    )


def _parse_motion_localization(data: dict[str, object]) -> StageConfig[MotionLocalizationConfig]:
    """Прочитать конфигурацию стадии motion_localization."""
    from ..stages.motion_localization import MOTION_LOCALIZER_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("motion_localization", {})),
        stage_name="motion_localization",
        config_classes=MOTION_LOCALIZER_CONFIG_CLASSES,
    )


def _parse_candidate_formation(data: dict[str, object]) -> StageConfig[CandidateFormerConfig]:
    """Прочитать конфигурацию стадии candidate_formation."""
    from ..stages.candidate_formation import CANDIDATE_FORMER_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("candidate_formation", {})),
        stage_name="candidate_formation",
        config_classes=CANDIDATE_FORMER_CONFIG_CLASSES,
    )


def _parse_candidate_filtering(data: dict[str, object]) -> StageConfig[CandidateFilterConfig]:
    """Прочитать конфигурацию стадии candidate_filtering."""
    from ..stages.candidate_filtering import CANDIDATE_FILTER_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("candidate_filtering", {})),
        stage_name="candidate_filtering",
        config_classes=CANDIDATE_FILTER_CONFIG_CLASSES,
    )


def _parse_target_selection(data: dict[str, object]) -> StageConfig[TargetSelectionConfig]:
    """Прочитать конфигурацию стадии target_selection."""
    from ..stages.target_selection import TARGET_SELECTION_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("target_selection", {})),
        stage_name="target_selection",
        config_classes=TARGET_SELECTION_CONFIG_CLASSES,
    )


def _parse_target_tracking(data: dict[str, object]) -> StageConfig[TargetTrackerConfig]:
    """Прочитать конфигурацию стадии target_tracking."""
    from ..stages.target_tracking import TARGET_TRACKER_CONFIG_CLASSES

    return StageConfigParser.parse(
        section=dict(data.get("target_tracking", {})),
        stage_name="target_tracking",
        config_classes=TARGET_TRACKER_CONFIG_CLASSES,
    )


def _parse_target_recovery(data: dict[str, object]) -> TargetRecoveryConfig:
    """Прочитать конфигурацию стадии target_recovery."""
    from ..stages.target_recovery import TARGET_RECOVERER_CONFIG_CLASSES

    section = dict(data.get("target_recovery", {}))
    stage_config = StageConfigParser.parse(
        section=section,
        stage_name="target_recovery",
        config_classes=TARGET_RECOVERER_CONFIG_CLASSES,
    )

    return TargetRecoveryConfig(
        enabled=stage_config.enabled,
        operations=stage_config.operations,
        min_lost_frames=_read_int(section, "min_lost_frames", default=5),
        confirm_frames=_read_int(section, "confirm_frames", default=3),
        recovery_window_frames=_read_int(section, "recovery_window_frames", default=30),
    )


def _parse_visualization(data: dict[str, object]) -> VisualizationConfig:
    """Прочитать конфигурацию отрисовки."""
    return VisualizationConfig.from_mapping(dict(data.get("visualization", {})))


def _parse_neural(data: dict[str, object]) -> NeuralConfig | None:
    """Прочитать конфигурацию нейросетевого контура."""
    neural_section = data.get("neural")

    if not isinstance(neural_section, dict) or not neural_section:
        return None

    return NeuralConfig.from_mapping(dict(neural_section))


def _inject_neural_config(
    target_tracking: StageConfig[TargetTrackerConfig],
    neural: NeuralConfig | None,
    path: Path,
) -> StageConfig[TargetTrackerConfig]:
    """Передать NeuralConfig в YOLO-tracker operation."""
    if not target_tracking.operations:
        return target_tracking

    from ..stages.target_tracking.operations import YoloTargetTrackerConfig

    patched_operations: list[TargetTrackerConfig] = []

    for operation in target_tracking.operations:
        if isinstance(operation, YoloTargetTrackerConfig):
            if neural is None:
                raise RuntimeError(
                    f"Preset {path} uses YOLO target tracker, but section [neural] is missing."
                )
            patched_operations.append(replace(operation, neural_config=neural))
        else:
            patched_operations.append(operation)

    return StageConfig(
        enabled=target_tracking.enabled,
        operations=tuple(patched_operations),
    )


def _read_int(section: dict[str, object], key: str, default: int) -> int:
    """Прочитать целочисленное поле секции."""
    value = section.get(key, default)

    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"target_recovery.{key} must be integer.")

    return value


def _build_preset_record(path: Path) -> _PresetRecord:
    """Создать полное описание пресета из одного TOML-файла."""
    data = _read_toml(path)
    preset_name, presentation = _parse_meta(path=path, data=data)
    neural = _parse_neural(data)
    target_tracking = _inject_neural_config(
        target_tracking=_parse_target_tracking(data),
        neural=neural,
        path=path,
    )

    return _PresetRecord(
        preset=TrackerPreset(
            name=preset_name,
            frame_preprocessing=_parse_frame_preprocessing(data),
            frame_stabilization=_parse_frame_stabilization(data),
            motion_localization=_parse_motion_localization(data),
            candidate_formation=_parse_candidate_formation(data),
            candidate_filtering=_parse_candidate_filtering(data),
            target_selection=_parse_target_selection(data),
            target_tracking=target_tracking,
            target_recovery=_parse_target_recovery(data),
            visualization=_parse_visualization(data),
            pipeline_kind=_parse_pipeline_kind(data=data, path=path),
            neural=neural,
        ),
        presentation=presentation,
        file_path=path,
    )


@lru_cache(maxsize=1)
def _load_preset_index() -> dict[str, _PresetIndexRecord]:
    """Прочитать список TOML-пресетов без сборки тяжёлых stage-конфигов."""
    if not PRESETS_DIR.exists():
        raise RuntimeError(f"Preset directory was not found: {PRESETS_DIR}")

    index: dict[str, _PresetIndexRecord] = {}

    for path in sorted(PRESETS_DIR.glob("*.toml")):
        data = _read_toml(path)
        if "meta" not in data:
            continue

        preset_name, presentation = _parse_meta(path=path, data=data)

        if preset_name in index:
            raise RuntimeError(f"Preset name {preset_name!r} is duplicated in {path.name}.")

        index[preset_name] = _PresetIndexRecord(
            name=preset_name,
            presentation=presentation,
            file_path=path,
        )

    if not index:
        raise RuntimeError(f"No TOML presets were found in {PRESETS_DIR}.")

    return index


@lru_cache(maxsize=None)
def _load_preset_record(name: str) -> _PresetRecord:
    """Загрузить и собрать один пресет по имени."""
    index = _load_preset_index()
    record = index[name]

    return _build_preset_record(record.file_path)


def _resolve_preset_name(name: str) -> str:
    """Привести имя пресета к реально существующему варианту."""
    requested = (name or "").strip()
    index = _load_preset_index()

    if requested in index:
        return requested

    if DEFAULT_PRESET_NAME in index:
        return DEFAULT_PRESET_NAME

    return next(iter(index))


def get_available_preset_names() -> tuple[str, ...]:
    """Вернуть список доступных пресетов в порядке загрузки."""
    index = _load_preset_index()
    ordered_names: list[str] = []

    if DEFAULT_PRESET_NAME in index:
        ordered_names.append(DEFAULT_PRESET_NAME)

    ordered_names.extend(name for name in index.keys() if name != DEFAULT_PRESET_NAME)

    return tuple(ordered_names)


def build_preset(name: str) -> TrackerPreset:
    """Вернуть полную конфигурацию пресета по имени."""
    preset_name = _resolve_preset_name(name)

    return deepcopy(_load_preset_record(preset_name).preset)


def get_preset_presentation(name: str) -> PresetPresentation:
    """Вернуть описание пресета для интерфейса."""
    preset_name = _resolve_preset_name(name)

    return _load_preset_index()[preset_name].presentation


# Готовый список имён для GUI и CLI без тяжёлой сборки stage-конфигов.
AVAILABLE_PRESETS = get_available_preset_names()
