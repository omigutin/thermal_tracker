from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar

from .stage_config import StageConfig


OperationConfigT = TypeVar("OperationConfigT")


class OperationConfigFactory(Protocol[OperationConfigT]):
    """Контракт класса конфигурации атомарной операции стадии."""

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> OperationConfigT:
        """Создать конфигурацию операции из сырых значений пресета."""
        ...


class StageConfigParser:
    """Парсер TOML-секции стадии с упорядоченным списком операций."""

    @classmethod
    def parse(
        cls,
        section: dict[str, object],
        stage_name: str,
        config_classes: Mapping[str, OperationConfigFactory[OperationConfigT]],
    ) -> StageConfig[OperationConfigT]:
        """Преобразовать TOML-секцию стадии в строгую конфигурацию."""

        if not config_classes:
            raise RuntimeError(f"Stage {stage_name!r} has no registered operation config classes.")
        if not section:
            return StageConfig(enabled=False, operations=())

        enabled = cls._parse_stage_enabled(section=section, stage_name=stage_name)
        operations = cls._parse_operations(section=section, stage_name=stage_name, config_classes=config_classes)

        return StageConfig(enabled=enabled, operations=operations)

    @staticmethod
    def _parse_stage_enabled(section: dict[str, object], stage_name: str) -> bool:
        """Прочитать признак включения всей стадии."""
        enabled = section.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RuntimeError(
                f"Stage {stage_name!r} field 'enabled' must be a boolean, "
                f"got {type(enabled).__name__}."
            )
        return enabled

    @classmethod
    def _parse_operations(
        cls,
        section: dict[str, object],
        stage_name: str,
        config_classes: Mapping[str, OperationConfigFactory[OperationConfigT]],
    ) -> tuple[OperationConfigT, ...]:
        """Прочитать список операций стадии."""

        raw_operations = section.get("operations")

        if raw_operations is None:
            cls._raise_if_legacy_operation_keys_used(section=section, stage_name=stage_name)
            return ()

        if not isinstance(raw_operations, list):
            raise RuntimeError(
                f"Stage {stage_name!r} field 'operations' must be an array "
                f"of TOML tables. Use [[{stage_name}.operations]]."
            )

        return tuple(
            cls._parse_operation_config(
                raw_operation=raw_operation,
                stage_name=stage_name,
                config_classes=config_classes,
            )
            for raw_operation in raw_operations
        )

    @classmethod
    def _parse_operation_config(
        cls,
        raw_operation: object,
        stage_name: str,
        config_classes: Mapping[str, OperationConfigFactory[OperationConfigT]],
    ) -> OperationConfigT:
        """Преобразовать TOML-описание одной операции в config-объект."""

        if not isinstance(raw_operation, dict):
            raise RuntimeError(
                f"Stage {stage_name!r} operation item must be a TOML table. "
                f"Use [[{stage_name}.operations]] with field 'type'."
            )

        raw_operation_data = dict(raw_operation)
        operation_type = cls._parse_operation_type(raw_operation_data=raw_operation_data, stage_name=stage_name)

        config_class = config_classes.get(operation_type)
        if config_class is None:
            raise RuntimeError(
                f"Unsupported operation type {operation_type!r} "
                f"for stage {stage_name!r}. "
                f"Available operation types: {tuple(sorted(config_classes))}."
            )

        return config_class.from_mapping(raw_operation_data)

    @staticmethod
    def _parse_operation_type(raw_operation_data: dict[str, object], stage_name: str) -> str:
        """Прочитать тип операции из TOML-описания."""

        operation_type = raw_operation_data.pop("type", None)

        if operation_type is None:
            raise RuntimeError(f"Stage {stage_name!r} operation item must contain 'type' field.")

        if not isinstance(operation_type, str):
            raise RuntimeError(
                f"Stage {stage_name!r} operation field 'type' must be a string, "
                f"got {type(operation_type).__name__}."
            )

        operation_type = operation_type.strip()
        if not operation_type:
            raise RuntimeError(f"Stage {stage_name!r} operation field 'type' must not be empty.")

        return operation_type

    @staticmethod
    def _raise_if_legacy_operation_keys_used(section: dict[str, object], stage_name: str) -> None:
        """Проверить, что не используется старый формат списка операций."""

        legacy_keys = ("filters", "methods")
        used_legacy_keys = tuple(key for key in legacy_keys if key in section)

        if not used_legacy_keys:
            return

        raise RuntimeError(
            f"Stage {stage_name!r} uses legacy operation keys "
            f"{used_legacy_keys}. Use [[{stage_name}.operations]] instead."
        )
