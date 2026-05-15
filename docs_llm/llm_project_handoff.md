# LLM Project Handoff: Thermal Tracker refactoring

## Назначение документа

Этот файл фиксирует текущее состояние рефакторинга, чтобы LLM могла безопасно продолжить рутинные задачи: аудит импортов, тесты, документацию и простые исправления.

Перед работой обязательно прочитать:

1. `docs_llm/llm_global_rules.md`
2. `docs_llm/llm_project_context.md`
3. `docs_llm/llm_project_architecture.md`
4. `docs_llm/llm_project_handoff.md`
5. `docs_llm/llm_project_tasks.md`

---

## Текущий статус

Проект находится в процессе рефакторинга stage/preset-архитектуры.

Вероятно, проект сейчас не запускается полностью из-за несинхронизированных импортов, старых путей, старых имён классов и временно оставленных `_OLD.py` файлов.

Это ожидаемое состояние.

Главная цель ближайшего этапа — не перепроектировать архитектуру, а аккуратно проверить импорты, синхронизировать новые пути, написать тесты на parsing, обновить документацию и не сломать принятые архитектурные решения.

---

## Уже принятые решения

- Stage-директории не нумеровать.
- Порядок выполнения задаётся через `pipeline.stage_order`.
- Новый preset package: `src/thermal_tracker/core/preset/`.
- `PresetFieldReader` остаётся в `core/preset/field_reader.py`.
- Не переименовывать `PresetFieldReader` в `ConfigFieldReader`.
- Общий `StageConfig` должен жить в `core/stages/config/`.
- Использовать TOML preset v2 через `[stages.<stage_name>]`.
- Не добавлять `VisualizationConfig` в root `Preset`.
- Не добавлять отдельный `NeuralConfig` в root `Preset`.
- `StagePreset.name`, не `StagePreset.id`.
- Файлы `_OLD.py` игнорировать.

---

## Текущие риски

1. Много импортов может ссылаться на старые пути:
   - `core/config/stage_config.py`
   - `core/config/stage_config_parser.py`
   - `core/config/preset_field_reader.py`
   - старый `core/config/preset.py`

2. В проекте могут остаться старые имена:
   - `TrackSnapshot`
   - `BaseSingleTargetTracker`
   - `ClickTargetSelector`
   - `YOLO_TRACK`
   - старые section names в TOML.

3. `thermal_tracker/__init__.py` может тянуть много алиасов и ломать isolated import-check.

4. Стадии `target_selection`, `target_tracking`, `target_recovery` были активно переработаны и могут содержать несинхронные импорты.

5. `src/thermal_tracker/core/config/model_config.py` выглядит как старый compatibility shim. Удалять нельзя без проверки импортов.

---

## Что нельзя делать без отдельного согласования

- Не менять архитектуру preset v2.
- Не нумеровать директории стадий.
- Не переносить визуализацию в core preset.
- Не возвращать `NeuralConfig` в root `Preset`.
- Не менять публичный API managers/factories без плана.
- Не удалять `_OLD.py` файлы.
- Не удалять `model_config.py` без поиска импортов.
- Не добавлять новые зависимости.
- Не менять `poetry.lock`.
- Не запускать Git-команды.
- Не делать массовые переименования без согласования.

---

## Рекомендуемый порядок технической стабилизации

1. Проверить текущие файлы `core/preset/`.
2. Проверить текущие файлы `core/stages/config/`.
3. Синхронизировать импорты `StageConfig`.
4. Синхронизировать импорты `StageConfigParser`.
5. Синхронизировать импорты `PresetFieldReader`.
6. Проверить `stage_registry.py`.
7. Проверить все stage-level `config.py`.
8. Проверить все `manager.py`.
9. Проверить все `factory.py`.
10. Запустить compileall.
11. Написать минимальные tests.
12. Запустить pytest.
13. Только потом чистить старые compatibility-файлы.
