from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresetMeta:
    """Человеческое описание пресета."""

    name: str
    title: str
    tooltip: str
    description: str

    def __post_init__(self) -> None:
        """Проверить обязательные поля описания пресета."""
        if not self.name.strip():
            raise ValueError("preset meta name must not be empty.")


@dataclass(frozen=True, slots=True)
class StagePreset:
    """Конфигурация одного экземпляра стадии внутри pipeline."""

    name: str
    stage_type: str
    enabled: bool
    config: object

    def __post_init__(self) -> None:
        """Проверить имя и тип экземпляра стадии."""
        if not self.name.strip():
            raise ValueError("stage name must not be empty.")

        if not self.stage_type.strip():
            raise ValueError("stage type must not be empty.")


@dataclass(frozen=True, slots=True)
class PresetPipeline:
    """Упорядоченный набор стадий обработки."""

    kind: str
    stages: tuple[StagePreset, ...]

    def __post_init__(self) -> None:
        """Проверить корректность pipeline."""
        if not self.kind.strip():
            raise ValueError("pipeline kind must not be empty.")

        stage_names = tuple(stage.name for stage in self.stages)

        if len(stage_names) != len(set(stage_names)):
            raise ValueError("pipeline stage names must be unique.")

    @property
    def enabled_stages(self) -> tuple[StagePreset, ...]:
        """Вернуть только включённые стадии в порядке выполнения."""
        return tuple(stage for stage in self.stages if stage.enabled)

    def get_stage(self, name: str) -> StagePreset | None:
        """Вернуть стадию по имени экземпляра."""
        for stage in self.stages:
            if stage.name == name:
                return stage

        return None

    def require_stage(self, name: str) -> StagePreset:
        """Вернуть стадию по имени или выбросить ошибку."""
        stage = self.get_stage(name)

        if stage is None:
            raise KeyError(f"Pipeline stage {name!r} was not found.")

        return stage

    def get_stages_by_type(self, stage_type: str) -> tuple[StagePreset, ...]:
        """Вернуть все стадии указанного типа в порядке выполнения."""
        return tuple(stage for stage in self.stages if stage.stage_type == stage_type)


@dataclass(frozen=True, slots=True)
class Preset:
    """Описание одного сценария обработки thermal tracker."""

    meta: PresetMeta
    pipeline: PresetPipeline

    @property
    def name(self) -> str:
        """Вернуть системное имя пресета."""
        return self.meta.name
