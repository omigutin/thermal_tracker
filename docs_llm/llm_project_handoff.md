# LLM Project Handoff: Thermal Tracker refactoring

## Назначение документа

Этот файл фиксирует текущее состояние рефакторинга, чтобы LLM могла безопасно продолжить работу после остановки, лимитов или смены модели.

Перед работой обязательно прочитать:

1. `docs_llm/llm_index.md`
2. `docs_llm/llm_global_rules.md`
3. `docs_llm/llm_project_context.md`
4. `docs_llm/llm_project_architecture.md`
5. `docs_llm/llm_project_handoff.md`
6. `docs_llm/llm_project_tasks.md`
7. `docs_llm/llm_worklog.md`, если файл существует

---

## Текущий статус

Проект находится в процессе рефакторинга stage/preset-архитектуры.

Предыдущая LLM выполнила тестовый запуск и остановилась из-за лимитов. Она успела:

- провести аудит части scope;
- исправить импорт `TargetRecoveryConfig` в `core/preset/parser.py`;
- удалить битый alias `processing_stages` из корневого `thermal_tracker/__init__.py`;
- добавить тесты на parsing-related сущности;
- обновить часть `docs_llm`;
- создать `docs_llm/llm_worklog.md`.

Перед продолжением обязательно проверить фактическое состояние Git:

```bash
git status
git log --oneline -n 10
```

если текущий task-файл разрешает локальные Git-команды.

---

## Важное уточнение после тестового запуска LLM

Документы были обновлены после первого тестового автономного запуска.

Выявленные проблемы предыдущих инструкций:

1. Не хватало явного автономного режима.
2. Не хватало явного разрешения локальных Git-коммитов в task-файле.
3. Был конфликт путей `field_reader.py` vs `preset_field_reader.py`.
4. `llm_worklog.md` не был указан в главном index.
5. Не было явного запрета коммитить `outputs/`, sandbox, cache-файлы и временные копии проекта.
6. Не была описана процедура продолжения после остановки LLM из-за лимитов.

Эти проблемы исправлены в актуальной версии документов.

---

## Уже принятые архитектурные решения

- Stage-директории не нумеровать.
- Порядок выполнения задаётся через `pipeline.stage_order`.
- Новый preset package: `src/thermal_tracker/core/preset/`.
- `PresetFieldReader` остаётся в `core/preset/field_reader.py`.
- Не переименовывать `PresetFieldReader` в `ConfigFieldReader`.
- Общий `StageConfig` живёт в `core/stages/config/`.
- Использовать TOML preset v2 через `[stages.<stage_name>]`.
- Не добавлять `VisualizationConfig` в root `Preset`.
- Не добавлять отдельный `NeuralConfig` в root `Preset`.
- Использовать `StagePreset.name`, не `StagePreset.id`.
- Файлы `_OLD.py` и `__OLD.py` игнорировать.

---

## Известные проблемы, которые нужно учитывать

### 1. Возможные незакоммиченные изменения предыдущей LLM

Предыдущая LLM заявила, что собиралась сделать коммит, но остановилась из-за лимитов.

Следующей LLM нужно сначала проверить:

```bash
git status
git log --oneline -n 10
```

Если изменения уже есть и соответствуют task scope, нужно зафиксировать их отдельным локальным коммитом.

### 2. Возможный конфликт имени файла `PresetFieldReader`

Актуальное архитектурное решение:

```text
src/thermal_tracker/core/preset/field_reader.py
class PresetFieldReader
```

Если в проекте создан `preset_field_reader.py`, не принимать это автоматически как новую архитектуру. Нужно привести код и документы к `field_reader.py`, если пользователь не решил иначе.

### 3. Старый публичный API `core/config/`

`core/config/__init__.py` может экспортировать не всё, что ещё импортируют старые потребители.

Возможные старые импорты:

- `AVAILABLE_PRESETS`;
- `build_preset`;
- `get_preset_presentation`;
- `TrackerPreset`;
- `DEFAULT_PRESET_NAME`;
- `PROJECT_ROOT`;
- `load_app_config`;
- `NeuralConfig`;
- `VisualizationConfig`;
- `ClickSelectionConfig`;
- `PresetFieldReader`.

Это отдельная большая задача. Не чинить массово без отдельного scope.

### 4. TOML-пресеты в старом формате

Многие `presets/*.toml` могут использовать старый формат:

- `[meta]` с `id` вместо `name`;
- секции `[frame_preprocessing]` вместо `[stages.<stage_name>]`;
- отсутствие `pipeline.stage_order`;
- `[visualization]` в корне.

Миграция пресетов — отдельная задача.

### 5. `TargetRecoveryConfig.stage` default factory

Возможный латентный баг: default factory может создавать `StageConfig(enabled=True, operations=())`, что нарушает инвариант `StageConfig`.

Исправлять отдельной точечной задачей, если текущий task разрешает.

### 6. `filter_type` vs `operation_type` в `candidate_filtering`

Возможное расхождение с архитектурным правилом.

Переименование может затронуть публичные имена и mappings. Не делать без отдельного разрешения.

### 7. Stale `__pycache__`

Перед проверками на Windows нужно удалить `__pycache__`.

---

## Что нельзя делать без отдельного разрешения

- Менять архитектуру preset v2.
- Нумеровать директории стадий.
- Переносить визуализацию в core preset.
- Возвращать `NeuralConfig` в root `Preset`.
- Менять публичный API managers/factories без плана.
- Удалять `_OLD.py` и `__OLD.py`.
- Удалять `preset_OLD.py`, `preset_loader_OLD.py`, `click_target_selector__OLD.py` без поиска импортов.
- Добавлять новые зависимости.
- Менять `poetry.lock`.
- Делать массовые переименования.
- Коммитить временные каталоги, cache-файлы, outputs или sandbox-копии.
- Выполнять `git push`.

---

## Рекомендуемый следующий порядок

1. Проверить `git status` и последние коммиты.
2. Если предыдущие изменения не закоммичены — проверить scope и сделать локальный коммит.
3. Удалить `__pycache__`.
4. Запустить `python -m compileall src/thermal_tracker/core`.
5. Запустить `pytest` по добавленным тестам.
6. Исправить только ошибки в рамках текущей архитектуры.
7. Обновить `llm_project_handoff.md`, `llm_project_tasks.md`, `llm_worklog.md`.
8. Сделать локальный коммит этапа, если коммиты разрешены task-файлом.
