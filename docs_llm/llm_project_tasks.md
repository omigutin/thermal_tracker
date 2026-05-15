# LLM Project Tasks: Import audit and preset parser stabilization

## Назначение документа

Это текущая задача для LLM. Файл можно перезаписывать после завершения этапа.

Перед выполнением прочитать:

1. `docs_llm/llm_global_rules.md`
2. `docs_llm/llm_project_context.md`
3. `docs_llm/llm_project_architecture.md`
4. `docs_llm/llm_project_handoff.md`
5. `docs_llm/llm_project_tasks.md`

---

## Главная цель этапа

Провести безопасный аудит после stage/preset refactoring.

Сначала нужен отчёт без правок. Код менять только после подтверждения пользователя.

---

## Область проверки

Проверять:

```text
src/thermal_tracker/core/preset/
src/thermal_tracker/core/stages/
src/thermal_tracker/core/config/
presets/
docs_llm/
```

Игнорировать:

```text
*_OLD.py
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
.git/
artifacts/
videos/
datasets/
weights/
```

---

## Этап 1. Аудит без изменений

Не менять файлы.

Нужно найти:

1. Битые импорты.
2. Импорты из файлов `_OLD.py`.
3. Старые импорты:
   - `core.config.stage_config`
   - `core.config.stage_config_parser`
   - `core.config.preset_field_reader`
4. Старые имена классов и сущностей:
   - `TrackSnapshot`
   - `BaseSingleTargetTracker`
   - `ClickTargetSelector`
   - `YOLO_TRACK`
   - `NeuralConfig` в root Preset
   - `VisualizationConfig` в core preset
5. Абсолютные импорты внутри `thermal_tracker`, которые можно заменить на относительные без риска.
6. Циклические импорты.
7. Места, где runtime pipeline работает с raw dict.
8. Места, где manager/factory знает TOML или строковый operation type.
9. Места, где используется старый TOML-формат.
10. Несинхронные exports в `__init__.py`.

---

## Формат отчёта

Вернуть отчёт:

```text
Аудит:
- ...

Найденные проблемы:
1. ...

Группы проблем:
- import path issues:
- obsolete names:
- wrong layer dependencies:
- preset format issues:
- tests missing:

Предлагаемый порядок исправлений:
1. ...
2. ...
3. ...

Файлы, которые нужно менять:
- ...

Риски:
- ...

Проверки, которые предлагается запустить:
- ...
```

После отчёта остановиться и ждать подтверждения.

---

## Этап 2. Исправление импортов

Выполнять только после подтверждения пользователя.

Разрешено:

- исправлять пути импортов;
- обновлять `__init__.py`;
- менять старые имена на уже утверждённые;
- не менять бизнес-логику.

Запрещено:

- менять архитектуру;
- менять публичный API без отдельного плана;
- удалять файлы;
- трогать `_OLD.py`;
- добавлять зависимости;
- менять TOML-формат v2;
- менять `poetry.lock`;
- выполнять Git-команды.

---

## Этап 3. Минимальные тесты

После стабилизации импортов добавить тесты.

Минимальный набор:

```text
tests/thermal_tracker/core/preset/test_parser.py
tests/thermal_tracker/core/preset/test_field_reader.py
tests/thermal_tracker/core/preset/test_stage_registry.py
tests/thermal_tracker/core/stages/config/test_stage_config_parser.py
```

Что проверить:

1. `PresetFieldReader` читает int/float/bool/str/tuple.
2. `PresetFieldReader.ensure_empty()` падает на неизвестных полях.
3. `StageConfigParser` парсит enabled stage с operations.
4. `StageConfigParser` падает на unknown operation type.
5. `StageConfigParser` падает на missing operation `type`.
6. `StageConfigParser` падает на legacy keys `methods` / `filters`.
7. `PresetParser` парсит preset v2.
8. `PresetParser` падает, если `stage_order` ссылается на отсутствующую stage-секцию.
9. `PresetParser` падает, если есть stage-секция вне `stage_order`.
10. `target_recovery` парсит stage-level поля.

---

## Проверки

Запустить только доступные инструменты проекта.

Минимально:

```bash
python -m compileall src/thermal_tracker/core
python -m pytest tests/thermal_tracker/core/preset tests/thermal_tracker/core/stages/config
```

Если `pytest` не настроен или тестов ещё нет, честно указать это в отчёте.

---

## Итоговый отчёт после изменений

После любых изменений вернуть:

```text
Итог:
- Что изменено:
- Почему так:
- Затронутые файлы:
- Проверки выполнены:
- Проверки не выполнены:
- Остаточные риски:
- Следующий рекомендуемый шаг:
```
