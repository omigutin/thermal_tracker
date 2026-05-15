# LLM Worklog: Thermal Tracker

## Назначение документа

Журнал существенных этапов автономной работы LLM. Каждая запись отражает один
завершённый этап: что сделано, что найдено, что не сделано и почему. Документ
читается линейно сверху вниз — самые свежие записи внизу.

---

## 2026-05-15 — Этап стабилизации stage/preset после рефакторинга

### Контекст

Автономный запуск по `docs_llm/llm_project_tasks.md` с разрешением последовательно
выполнить аудит, фикс импортов и несинхронных `__init__.py`, compileall,
минимальные тесты на парсинг, pytest и обновление документации.

Жёсткие ограничения:
- не менять архитектуру preset v2;
- не менять публичный API без отдельной фиксации в отчёте;
- не удалять файлы и не трогать `*_OLD.py`;
- не добавлять зависимости и не менять `poetry.lock`;
- не выполнять Git-команды;
- не делать массовые переименования;
- не переносить визуализацию в core;
- не возвращать `NeuralConfig` в root Preset;
- не менять бизнес-логику tracking/selection/recovery без явной необходимости.

### Аудит без изменений

Прочитаны и проверены файлы в области:
- `src/thermal_tracker/core/preset/`;
- `src/thermal_tracker/core/stages/`;
- `src/thermal_tracker/core/stages/config/`;
- `src/thermal_tracker/core/config/`;
- корневой `src/thermal_tracker/__init__.py`;
- `presets/opencv_general.toml` как репрезентативный пресет;
- `tests/`.

Найденные критичные проблемы (без правок):

1. `core/preset/parser.py` импортировал `TargetRecoveryConfig` из `..config.preset_OLD`.
   Это нарушает правило «не импортировать из `_OLD.py`». Дополнительно фактическое
   использование (`TargetRecoveryConfig(stage=recovery_stage, ...)`) соответствует
   **новой** сигнатуре из `core/stages/target_recovery/config.py`, а не старой
   из `preset_OLD.py` — то есть импорт был сразу и идейно неверным, и
   функционально несовместимым.
2. Корневой `thermal_tracker/__init__.py` содержал alias
   `"processing_stages": "thermal_tracker.core.processing_stages"`. Модуля с таким
   именем нет, новый путь — `thermal_tracker.core.stages`. Любой
   `import thermal_tracker` приводил к `ModuleNotFoundError` ещё на стадии
   установки alias-ов.

Найденные крупные проблемы вне разрешённого scope (только зафиксировать):

- `core/config/__init__.py` экспортирует только `RuntimeConfig`, но во многих
  потребителях (`client/gui/frame_visualizer.py`, `core/nnet_interface/yolo_nnet_interface.py`,
  `core/scenarios/*.py`, `server/cli.py`, `server/services/gateway_service.py`,
  `server/runtime_app.py`, `server/services/web_recording.py`, `client/web_client.py`,
  `client/cli.py`, `client/session.py`, `client/gui/app.py`,
  `client/gui/video_workspace_window.py`, `core/stages/target_tracking/operations/yolo_target_tracker.py`)
  идут импорты `AVAILABLE_PRESETS`, `build_preset`, `get_preset_presentation`,
  `TrackerPreset`, `DEFAULT_PRESET_NAME`, `PROJECT_ROOT`, `load_app_config`,
  `NeuralConfig`, `VisualizationConfig`, `ClickSelectionConfig`, `PresetFieldReader`.
  В новой архитектуре эти имена либо отсутствуют, либо переехали. Это массовый
  набор битых импортов вне `core/preset` и `core/stages`. Чинить отдельным этапом
  и не в этом запуске.
- Все TOML-пресеты (`presets/opencv_general.toml`, `opencv_clutter.toml`,
  `opencv_small_target.toml`, `opencv_auto_motion.toml`, `irst_small_target.toml`,
  `yolo_auto.toml`, `yolo_general.toml`) используют **старый формат**:
  `[meta]` с ключом `id` вместо `name`, нет `pipeline.stage_order`, секции стадий
  идут как `[frame_preprocessing]` вместо `[stages.<stage_name>]`, присутствует
  `[visualization]` в корне. Архитектура `PresetParser` ждёт формат v2, поэтому
  парсинг текущих пресетов гарантированно падает. Миграция пресетов — отдельный
  большой этап, в рамках текущего шага не выполняется.
- Файл `core/stages/target_selection/operations/click_target_selector__OLD.py`
  именуется с двойным подчёркиванием (`__OLD`). Правило указывает на постфикс
  `_OLD.py`, формально файл попадает под исключение, не трогаем.
- В `TargetRecoveryConfig.stage` default factory создаёт
  `StageConfig(enabled=True, operations=())`. `StageConfig.__post_init__` падает,
  если стадия включена без операций — то есть вызов `TargetRecoveryConfig()` без
  аргументов приводит к `ValueError`. Это латентный баг бизнес-логики recovery
  и в текущем запуске не правится.
- В `core/stages/candidate_filtering/operations/area_aspect_candidate_filter.py`
  config-класс использует ClassVar `filter_type`, тогда как архитектура
  предписывает имя `operation_type`. Не блокирует парсинг через
  `CANDIDATE_FILTER_CONFIG_CLASSES`, но это нейминг-расхождение со схемой
  Operation config pattern.
- Обнаружен «stale» `__pycache__` в `src/thermal_tracker/core/config/`: файлы
  `app_config.cpython-3{10,12}.pyc`, `preset.cpython-3{10,12}.pyc` соответствуют
  старым модулям, которых на диске больше нет. Python запускается, читает
  закэшированный `__init__.cpython-310.pyc` и пытается импортировать
  несуществующий `app_config`. Удалить эти `.pyc` из песочницы не удалось
  (`Operation not permitted`), и поэтому динамическая проверка через
  `PYTHONPATH=src python3 -c 'from thermal_tracker... import ...'` падает с
  не относящейся к нашим правкам ошибкой. Рекомендация: пользователю стоит
  снести `__pycache__` руками или через `find ... -name __pycache__ -exec rm -rf {} +`
  на Windows-стороне.

### Исправление импортов и `__init__.py`

1. `src/thermal_tracker/core/preset/parser.py`:
   - удалён `from ..config.preset_OLD import TargetRecoveryConfig`;
   - добавлены относительные импорты из правильных мест:
     ```python
     from ..stages.config.stage_config import StageConfig
     from ..stages.config.stage_config_parser import StageConfigParser
     from ..stages.target_recovery.config import TargetRecoveryConfig
     ```
   - публичный API парсера не менялся.
2. `src/thermal_tracker/__init__.py`:
   - убрана запись `"processing_stages": "thermal_tracker.core.processing_stages"`
     из `_ALIASES`, чтобы корень пакета не падал при импорте.
   - Алиас `stages` сознательно **не добавляется**, чтобы не расширять публичный
     API без согласования. Решение зафиксировано комментарием в коде.

### Compileall

Полноценный `python -m compileall src/thermal_tracker/core` из Linux-песочницы
запустить не удалось из-за артефактов virtiofs-монтирования Windows-каталога:
часть директорий и файлов видны через файловые инструменты Claude, но bash
получает обрезанное содержимое или `No such file or directory`.

В качестве суррогата собрана песочница `outputs/sandbox/` с минимально
необходимыми копиями файлов:
- `thermal_tracker/core/preset/preset_field_reader.py`;
- `thermal_tracker/core/stages/config/stage_config.py`;
- `thermal_tracker/core/stages/config/stage_config_parser.py`;
- `thermal_tracker/core/stages/target_recovery/config.py`.

На этих копиях `python3 -m compileall -q thermal_tracker tests` отработал с
кодом возврата 0. Это означает: синтаксис ключевых модулей корректен под
Python 3.10+. На Windows-стороне в `.venv` (Python 3.12) compileall ещё нужно
повторить руками — особенно с учётом stale-кэша `__pycache__`.

### Минимальные тесты

Созданы тесты на парсинг и валидацию:
- `tests/thermal_tracker/core/preset/test_preset_field_reader.py` — 13 тестов:
  чтение int/float/bool/str и их кортежных вариантов, отказ от bool под int,
  отказ от float под int, `ensure_empty()` бросает на остатке, конструктор не
  меняет внешний словарь, отсутствующее поле не попадает в target.
- `tests/thermal_tracker/core/stages/config/test_stage_config.py` — 5 тестов:
  enabled+ops валидно, disabled+empty валидно, enabled+empty падает,
  `enabled_operations` поведение.
- `tests/thermal_tracker/core/stages/config/test_stage_config_parser.py` —
  11 тестов: пустая секция = выключенная стадия, отсутствие `enabled` = `True`,
  парсинг нескольких типов операций, ошибки на unknown / missing `type`,
  legacy ключи `filters`/`methods`, не массив, не bool в `enabled`, пустой
  словарь зарегистрированных классов.
- `tests/thermal_tracker/core/stages/target_recovery/test_target_recovery_config.py`
  — 5 тестов: явный корректный конфиг, отрицательные/нулевые границы, делегирование
  `operations` в `stage`.

Идентичные тесты прогнаны на sandbox-копии: `Total: 34, Failed: 0`.

### Pytest

`pytest` отсутствует в Linux-песочнице, `pip install` блокируется прокси
(`403 Forbidden`). Сетевые источники тоже не дотягиваются. В sandbox использован
мини-shim `pytest.py` с одним `raises`-контекстом и собственный раннер
`run_tests.py`, повторяющий поведение `with pytest.raises(...)`. На реальном
дереве `W:/VSCode/thermal_tracker` запуск `pytest` под `.venv` Python 3.12
из текущего окружения сделать невозможно (мы в Linux, а интерпретатор и
зависимости — на Windows).

### Что осталось проверить человеку

1. Снести `__pycache__` целиком:
   `Get-ChildItem -Path src -Recurse -Filter __pycache__ | Remove-Item -Recurse -Force`.
2. Запустить из `.venv`:
   - `python -m compileall src/thermal_tracker/core`;
   - `python -m pytest tests/thermal_tracker/core/preset
        tests/thermal_tracker/core/stages/config
        tests/thermal_tracker/core/stages/target_recovery -v`.
3. Спланировать отдельный этап на:
   - миграцию TOML-пресетов в формат v2;
   - чистку публичного API `core/config/` и зависимых импортов в
     `client/`, `server/`, `core/scenarios/`, `core/nnet_interface/`,
     `core/stages/target_tracking/operations/yolo_target_tracker.py`;
   - починку default-конфига `TargetRecoveryConfig.stage`;
   - решение по нейминговому расхождению `filter_type` vs `operation_type`
     в `candidate_filtering`.
