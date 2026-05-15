# LLM Project Architecture: Thermal Tracker stages and presets

## Назначение документа

Этот документ фиксирует текущую стабильную архитектуру stage/preset-слоя проекта `thermal_tracker`.

Если этот документ конфликтует со старым кодом, считать возможным, что код устарел. Перед изменениями нужно проверить фактическое состояние файлов и согласовать план с пользователем или действовать только в рамках текущего `llm_project_tasks.md`.

---

## Термины

### Preset

`Preset` — описание одного сценария обработки.

Preset хранит:

- метаданные;
- тип pipeline;
- порядок экземпляров стадий;
- настройки стадий.

Preset — внешний слой настройки, обычно описанный в TOML-файле.

### Config

`Config` — уже собранный строгий объект настроек.

Runtime-код должен работать с config-объектами, а не с raw dict.

### Stage type

`stage_type` — тип стадии, то есть какую задачу умеет решать стадия.

Примеры:

```text
frame_preprocessing
frame_stabilization
motion_localization
candidate_formation
candidate_filtering
target_selection
target_tracking
target_recovery
```

Stage type соответствует директории в:

```text
src/thermal_tracker/core/stages/
```

### Stage name

`stage_name` — имя конкретного экземпляра стадии в одном пресете.

Пример:

```toml
[pipeline]
stage_order = [
    "input_preprocessing",
    "camera_stabilization",
    "nn_preprocessing",
]

[stages.input_preprocessing]
type = "frame_preprocessing"

[stages.nn_preprocessing]
type = "frame_preprocessing"
```

`input_preprocessing` и `nn_preprocessing` — разные экземпляры одного stage type `frame_preprocessing`.

### Operation

`Operation` — атомарная операция внутри стадии.

---

## Stage-директории

Директории стадий не нумеруются.

Причина: директория стадии описывает stage type, а порядок выполнения задаётся в пресете через `pipeline.stage_order`.

```text
src/thermal_tracker/core/stages/
├── candidate_filtering
├── candidate_formation
├── config
├── frame_preprocessing
├── frame_stabilization
├── motion_localization
├── target_recovery
├── target_selection
└── target_tracking
```

Порядок выполнения по умолчанию для классического pipeline:

| Порядок | Stage name в пресете | Stage type | Назначение |
|---:|---|---|---|
| 1 | `input_preprocessing` | `frame_preprocessing` | Подготовка входного кадра |
| 2 | `camera_stabilization` | `frame_stabilization` | Оценка / компенсация движения камеры |
| 3 | `motion_localization` | `motion_localization` | Поиск областей движения |
| 4 | `candidate_formation` | `candidate_formation` | Сборка кандидатов |
| 5 | `candidate_filtering` | `candidate_filtering` | Фильтрация ложных кандидатов |
| 6 | `target_selection` | `target_selection` | Выбор цели |
| 7 | `target_tracking` | `target_tracking` | Сопровождение выбранной цели |
| 8 | `target_recovery` | `target_recovery` | Восстановление потерянной цели |

Фактический порядок всегда берётся из `pipeline.stage_order`.

---

## Preset package

Новый слой работы с пресетами:

```text
src/thermal_tracker/core/preset/
├── __init__.py
├── field_reader.py
├── parser.py
├── preset.py
└── stage_registry.py
```

### `preset.py`

Содержит только модели данных:

```text
Preset
PresetMeta
PresetPipeline
StagePreset
```

`Preset` не должен иметь фиксированных полей вида:

```text
frame_preprocessing
target_tracking
target_recovery
```

Вместо этого используется ordered pipeline:

```text
Preset
└── PresetPipeline
    └── tuple[StagePreset, ...]
```

### `field_reader.py`

Содержит `PresetFieldReader`.

`PresetFieldReader` читает типизированные поля из сырого описания пресета.

Он используется только на этапе:

```text
raw dict from TOML
→ from_mapping()
→ strict dataclass config
```

`PresetFieldReader` не должен использоваться в runtime pipeline, managers или factories.

Актуальный путь:

```text
src/thermal_tracker/core/preset/field_reader.py
```

Актуальный класс:

```text
PresetFieldReader
```

Не использовать путь:

```text
src/thermal_tracker/core/preset/preset_field_reader.py
```

если пользователь отдельно не изменит это архитектурное решение.

### `parser.py`

Содержит `PresetParser`.

`PresetParser` отвечает за:

- чтение `[meta]`;
- чтение `[pipeline]`;
- проверку `stage_order`;
- чтение `[stages.<stage_name>]`;
- выбор stage parser по `type`;
- сборку `Preset`.

`PresetParser` не должен знать параметры конкретных операций.

### `stage_registry.py`

Содержит registry связи:

```text
stage_type → operation config classes
```

Пример:

```text
candidate_filtering → CANDIDATE_FILTER_CONFIG_CLASSES
target_tracking     → TARGET_TRACKER_CONFIG_CLASSES
```

---

## Общие stage config-объекты

Общие конфиги стадий должны лежать здесь:

```text
src/thermal_tracker/core/stages/config/
├── __init__.py
├── stage_config.py
└── stage_config_parser.py
```

### `StageConfig`

`StageConfig` — общий контейнер конфигурации operation-stage.

Он хранит только:

```text
enabled: bool
operations: tuple[OperationConfigT, ...]
```

Он не знает конкретные стадии и бизнес-логику.

### `StageConfigParser`

`StageConfigParser` парсит одну stage-секцию в `StageConfig`.

Он делает:

```text
raw stage section
→ enabled
→ operations
→ operation type
→ operation config class
→ operation config object
→ StageConfig[OperationConfig]
```

Он не должен знать поля конкретных операций вроде `min_area`, `kernel`, `threshold`, `search_radius`.

---

## Формат TOML preset v2

```toml
[meta]
name = "opencv_general"
title = "OpenCV general"
tooltip = "Базовый OpenCV-пресет"
description = "Ручной выбор цели и сопровождение через template-point tracker."

[pipeline]
kind = "manual_click_classical"
stage_order = [
    "input_preprocessing",
    "camera_stabilization",
    "motion_localization",
    "candidate_formation",
    "candidate_filtering",
    "target_selection",
    "target_tracking",
    "target_recovery",
]

[stages.input_preprocessing]
type = "frame_preprocessing"
enabled = true

[[stages.input_preprocessing.operations]]
type = "resize"
enabled = true
width = 960
```

Правила:

- `[meta]` использует `name`, не `id`;
- `pipeline.stage_order` содержит stage name, не stage type;
- `stages.<stage_name>.type` содержит stage type;
- stage name должен быть уникален;
- секции из `[stages]`, которых нет в `stage_order`, считаются ошибкой;
- элементы из `stage_order`, для которых нет секции `[stages.<name>]`, считаются ошибкой.

---

## Stage package pattern

Для каждой стадии желательно использовать структуру:

```text
stage_name/
├── __init__.py
├── config.py
├── factory.py
├── manager.py
├── result.py       # если стадия возвращает собственный результат
├── type.py
└── operations/
    ├── __init__.py
    ├── base_*.py
    └── concrete_operation.py
```

Если стадии нужны вложенные алгоритмы, можно использовать подкаталоги внутри `operations/`.

Пример:

```text
target_selection/
└── operations/
    └── contrast_component/
        ├── __init__.py
        ├── contrast_component_target_selector.py
        ├── contrast_component_mask_builder.py
        └── ...
```

---

## Operation config pattern

Config-класс конкретной операции должен лежать рядом с runtime-классом операции.

В одном файле обычно находятся:

```text
ConcreteOperationConfig
ConcreteOperation
```

Config-класс должен иметь:

```text
operation_type: ClassVar[Enum]
enabled: bool = True
from_mapping(...)
__post_init__()
```

`from_mapping()` должен:

1. принять `dict[str, object]`;
2. создать `PresetFieldReader`;
3. прочитать только явно заданные поля;
4. вызвать `ensure_empty()`;
5. вернуть `cls(**kwargs)`.

Дефолты должны жить в dataclass-полях config-класса.

Нельзя дублировать дефолты внутри `from_mapping()`.

---

## Runtime operation pattern

Runtime-класс операции должен хранить config-объект.

Runtime-класс не должен дублировать поля config-класса.

---

## Factory pattern

Factory стадии:

- принимает operation config-объекты;
- пропускает `operation_config.enabled == False`;
- создаёт runtime-объект по типу config;
- не знает TOML;
- не работает с raw dict;
- не валидирует поля TOML;
- не знает `StageConfig`, если это не нужно.

Для небольшого числа операций допустим явный `if isinstance(...)`, чтобы не ломать типизацию IDE.

---

## Manager pattern

Manager стадии:

- принимает `StageConfig[OperationConfig]` или специальный stage-level config;
- проверяет `stage.enabled`;
- через factory создаёт runtime operations;
- хранит runtime operations в `tuple`;
- выполняет операции в порядке из config;
- не знает TOML;
- не знает строковые operation type;
- не парсит raw dict.

---

## Особый случай: target_recovery

`target_recovery` отличается от обычных operation-stage.

У неё есть stage-level параметры:

```text
min_lost_frames
confirm_frames
recovery_window_frames
```

Эти параметры не относятся к конкретной operation.

Поэтому `target_recovery/config.py` должен содержать:

```text
TargetRecovererConfig
TARGET_RECOVERER_CONFIG_CLASSES
TargetRecoveryConfig
```

`TargetRecoveryConfig` должен хранить:

```text
enabled
stage: StageConfig[TargetRecovererConfig]
min_lost_frames
confirm_frames
recovery_window_frames
```

---

## Визуализация и neural config

Визуализация не должна жить в core preset.

Отдельный `NeuralConfig` в модели `Preset` не нужен.

Если нейросеть используется в конкретной стадии, её настройки должны быть частью config-класса этой стадии или operation-класса.

---

## Deprecated / old files

Файлы с постфиксом `_OLD.py` и `__OLD.py` игнорировать.

LLM не должен:

- импортировать из них;
- исправлять их;
- включать их в тесты;
- учитывать их как актуальную архитектуру.
