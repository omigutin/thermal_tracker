from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..config.preset_OLD import TargetRecoveryConfig
from thermal_tracker.core.stages.config.stage_config import StageConfig
from thermal_tracker.core.stages.config.stage_config_parser import StageConfigParser
from .preset_field_reader import PresetFieldReader
from .preset_preset import Preset, PresetMeta, PresetPipeline, StagePreset
from .preset_stage_registry import StageRegistry, StageRegistryItem


class PresetParser:
    """Преобразует TOML-словарь пресета в объект Preset."""

    def parse(self, data: Mapping[str, object], source_path: Path | None = None) -> Preset:
        """Распарсить словарь TOML-пресета."""

        meta = self._parse_meta(data=data, source_path=source_path)
        pipeline_kind = self._parse_pipeline_kind(data)
        stage_order = self._parse_stage_order(data)
        stages_data = self._parse_stages_data(data)
        stage_presets = self._parse_stage_presets(stage_order=stage_order, stages_data=stages_data)

        return Preset(meta=meta, pipeline=PresetPipeline(kind=pipeline_kind, stages=stage_presets))

    def _parse_meta(self, data: Mapping[str, object], source_path: Path | None) -> PresetMeta:
        """Распарсить человекочитаемое описание пресета."""

        meta = self._get_mapping(data=data, key="meta", owner="preset", required=True)
        name = self._read_required_str(data=meta, key="name", owner="meta")
        title = self._read_optional_str(data=meta, key="title", default=name, owner="meta")
        tooltip = self._read_optional_str(data=meta, key="tooltip", default=f"Пресет {name}", owner="meta")
        description = self._read_optional_str(data=meta, key="description", default=tooltip, owner="meta")

        self._ensure_no_unknown_meta_fields(meta)

        return PresetMeta(name=name, title=title, tooltip=tooltip, description=description)

    def _parse_pipeline_kind(self, data: Mapping[str, object]) -> str:
        """Распарсить тип pipeline-сценария."""

        pipeline = self._get_mapping(data=data, key="pipeline", owner="preset", required=True)

        return self._read_optional_str(data=pipeline, key="kind", default="manual_click_classical", owner="pipeline")

    def _parse_stage_order(self, data: Mapping[str, object]) -> tuple[str, ...]:
        """Распарсить порядок экземпляров стадий."""

        pipeline = self._get_mapping(data=data, key="pipeline", owner="preset", required=True)

        raw_stage_order = pipeline.get("stage_order")
        if raw_stage_order is None:
            raise ValueError("pipeline.stage_order must be specified.")
        if not isinstance(raw_stage_order, list | tuple):
            raise TypeError(f"pipeline.stage_order must be array, got {type(raw_stage_order).__name__}.")

        stage_order: list[str] = []

        for raw_stage_name in raw_stage_order:
            if not isinstance(raw_stage_name, str):
                raise TypeError("pipeline.stage_order items must be string, got {type(raw_stage_name).__name__}.")
            stage_name = raw_stage_name.strip()
            if not stage_name:
                raise ValueError("pipeline.stage_order items must not be empty.")
            stage_order.append(stage_name)

        if not stage_order:
            raise ValueError("pipeline.stage_order must not be empty.")

        if len(stage_order) != len(set(stage_order)):
            raise ValueError("pipeline.stage_order must contain unique stage names.")

        return tuple(stage_order)

    def _parse_stages_data(self, data: Mapping[str, object]) -> dict[str, dict[str, object]]:
        """Распарсить таблицу stages из TOML-пресета."""
        raw_stages = self._get_mapping(data=data, key="stages", owner="preset", required=True)

        stages: dict[str, dict[str, object]] = {}

        for stage_name, stage_data in raw_stages.items():
            if not isinstance(stage_name, str) or not stage_name.strip():
                raise ValueError("stage name must be a non-empty string.")

            if not isinstance(stage_data, Mapping):
                raise TypeError(f"stages.{stage_name} must be table, got {type(stage_data).__name__}.")

            stages[stage_name.strip()] = dict(stage_data)

        return stages

    def _parse_stage_presets(
        self,
        stage_order: tuple[str, ...],
        stages_data: Mapping[str, dict[str, object]],
    ) -> tuple[StagePreset, ...]:
        """Распарсить стадии в порядке pipeline.stage_order."""
        self._validate_stage_order_matches_sections(stage_order=stage_order, stages_data=stages_data)

        stage_presets: list[StagePreset] = []

        for stage_name in stage_order:
            stage_data = dict(stages_data[stage_name])
            stage_type = self._pop_stage_type(stage_name=stage_name, stage_data=stage_data)
            registry_item = StageRegistry.get(stage_type)
            config = self._parse_stage_config(
                stage_name=stage_name,
                stage_type=stage_type,
                stage_data=stage_data,
                registry_item=registry_item,
            )

            stage_presets.append(
                StagePreset(
                    name=stage_name,
                    stage_type=stage_type,
                    enabled=self._resolve_stage_enabled(config),
                    config=config,
                )
            )

        return tuple(stage_presets)

    def _parse_stage_config(
        self,
        stage_name: str,
        stage_type: str,
        stage_data: dict[str, object],
        registry_item: StageRegistryItem,
    ) -> object:
        """Распарсить конфиг конкретного экземпляра стадии."""
        if stage_type == "target_recovery":
            return self._parse_target_recovery_config(
                stage_name=stage_name,
                stage_data=stage_data,
                registry_item=registry_item,
            )

        return StageConfigParser.parse(
            section=stage_data,
            stage_name=stage_name,
            config_classes=registry_item.config_classes,
        )

    def _parse_target_recovery_config(
        self,
        stage_name: str,
        stage_data: dict[str, object],
        registry_item: StageRegistryItem,
    ) -> TargetRecoveryConfig:
        """Распарсить конфиг стадии target_recovery с общими параметрами."""
        stage_section = {
            "enabled": stage_data.get("enabled", True),
            "operations": stage_data.get("operations", ()),
        }
        recovery_stage = StageConfigParser.parse(
            section=stage_section,
            stage_name=stage_name,
            config_classes=registry_item.config_classes,
        )

        reader_values = dict(stage_data)
        reader_values.pop("enabled", None)
        reader_values.pop("operations", None)

        kwargs: dict[str, object] = {
            "enabled": recovery_stage.enabled,
            "stage": recovery_stage,
        }
        reader = PresetFieldReader(owner=f"stages.{stage_name}", values=reader_values)
        reader.pop_int_to(kwargs, "min_lost_frames")
        reader.pop_int_to(kwargs, "confirm_frames")
        reader.pop_int_to(kwargs, "recovery_window_frames")
        reader.ensure_empty()

        return TargetRecoveryConfig(**kwargs)

    @staticmethod
    def _pop_stage_type(stage_name: str, stage_data: dict[str, object]) -> str:
        """Извлечь тип стадии из stage-секции."""
        raw_stage_type = stage_data.pop("type", None)

        if raw_stage_type is None:
            raise ValueError(f"stages.{stage_name}.type must be specified.")

        if not isinstance(raw_stage_type, str):
            raise TypeError(f"stages.{stage_name}.type must be string, got {type(raw_stage_type).__name__}.")

        stage_type = raw_stage_type.strip()

        if not stage_type:
            raise ValueError(f"stages.{stage_name}.type must not be empty.")

        return stage_type

    @staticmethod
    def _resolve_stage_enabled(config: object) -> bool:
        """Определить, включена ли стадия по её config-объекту."""
        if isinstance(config, StageConfig):
            return config.enabled

        if isinstance(config, TargetRecoveryConfig):
            return config.enabled

        raise TypeError(f"Unsupported stage config object: {type(config).__name__!r}.")

    @staticmethod
    def _validate_stage_order_matches_sections(
        stage_order: tuple[str, ...],
        stages_data: Mapping[str, object],
    ) -> None:
        """Проверить соответствие stage_order и секций stages."""
        ordered_names = set(stage_order)
        section_names = set(stages_data)

        missing_sections = tuple(sorted(ordered_names - section_names))
        if missing_sections:
            raise ValueError(
                f"pipeline.stage_order references missing stage sections: "
                f"{missing_sections}."
            )

        unused_sections = tuple(sorted(section_names - ordered_names))
        if unused_sections:
            raise ValueError(
                f"preset has stage sections not listed in pipeline.stage_order: "
                f"{unused_sections}."
            )

    @staticmethod
    def _get_mapping(
        data: Mapping[str, object],
        key: str,
        owner: str,
        *,
        required: bool,
    ) -> dict[str, object]:
        """Прочитать вложенную TOML-таблицу."""
        value = data.get(key)

        if value is None:
            if required:
                raise ValueError(f"{owner}.{key} section must be specified.")

            return {}

        if not isinstance(value, Mapping):
            raise TypeError(f"{owner}.{key} must be table, got {type(value).__name__}.")

        return dict(value)

    @staticmethod
    def _read_required_str(data: Mapping[str, object], key: str, owner: str) -> str:
        """Прочитать обязательную строку."""
        value = data.get(key)

        if value is None:
            raise ValueError(f"{owner}.{key} must be specified.")

        if not isinstance(value, str):
            raise TypeError(f"{owner}.{key} must be string, got {type(value).__name__}.")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{owner}.{key} must not be empty.")

        return normalized

    @staticmethod
    def _read_optional_str(
        data: Mapping[str, object],
        key: str,
        default: str,
        owner: str,
    ) -> str:
        """Прочитать необязательную строку."""
        value = data.get(key)

        if value is None:
            return default

        if not isinstance(value, str):
            raise TypeError(f"{owner}.{key} must be string, got {type(value).__name__}.")

        normalized = value.strip()

        if not normalized:
            return default

        return normalized

    @staticmethod
    def _ensure_no_unknown_meta_fields(meta: Mapping[str, object]) -> None:
        """Проверить, что в meta нет неизвестных полей."""

        allowed_fields = {"name", "title", "tooltip", "description"}
        unknown_fields = tuple(sorted(set(meta) - allowed_fields))

        if unknown_fields:
            raise ValueError(f"meta has unsupported fields: {unknown_fields}.")
