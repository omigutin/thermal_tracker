# LLM Project Tasks: Verify on Windows side + plan remaining cleanup

## Назначение документа

Это текущая задача для LLM. Файл можно перезаписывать после завершения этапа.

Перед выполнением прочитать:

1. `docs_llm/llm_global_rules.md`
2. `docs_llm/llm_project_context.md`
3. `docs_llm/llm_project_architecture.md`
4. `docs_llm/llm_project_handoff.md`
5. `docs_llm/llm_project_tasks.md`
6. `docs_llm/llm_worklog.md`

---

## Контекст

Предыдущий этап («аудит + минимальные импортные правки + тесты на парсинг»)
завершён. Подробности — в `llm_worklog.md` от 2026-05-15.

Что сделано:

- `core/preset/parser.py` — починен импорт `TargetRecoveryConfig`
  (теперь из `..stages.target_recovery.config`, не из `_OLD`).
- `src/thermal_tracker/__init__.py` — удалён битый alias `processing_stages`.
- Добавлены unit-тесты на `PresetFieldReader`, `StageConfig`,
  `StageConfigParser`, `TargetRecoveryConfig`.

Что **не** удалось запустить из текущего окружения:

- `python -m compileall` на реальном дереве — мешает stale `__pycache__` и
  артефакты virtiofs;
- `pytest` на реальном дереве — `pytest` есть только в Windows `.venv`,
  доступа к нему из Linux нет.

---

## Главная цель ближайшего этапа

Перенести проверки на Windows-сторону и подготовить отдельные согласованные
задачи на оставшиеся системные проблемы.

---

## Этап 1. Очистка кэша и базовые проверки (Windows, ручной запуск)

Выполнить пользователю в PowerShell в корне проекта (`W:\VSCode\thermal_tracker`)
с активированным `.venv`:

```powershell
Get-ChildItem -Path src, tests -Recurse -Filter __pycache__ -Force `
    | Remove-Item -Recurse -Force
python -m compileall src/thermal_tracker/core
python -m pytest tests/thermal_tracker/core/preset `
    tests/thermal_tracker/core/stages/config `
    tests/thermal_tracker/core/stages/target_recovery -v
```

Ожидаемый результат:

- `compileall` отрабатывает без ошибок;
- pytest показывает 34 пройденных теста.

Если что-то падает — зафиксировать вывод и передать следующему этапу.

---

## Этап 2. План на миграцию TOML-пресетов в формат v2

Все файлы в `presets/*.toml` пока в старом формате. `PresetParser` их не
разберёт. Этап выполняется отдельной задачей с прямым согласованием
пользователя:

1. Сформировать пример миграции одного пресета (`opencv_general.toml`) в формат
   v2:
   - `[meta]` использует `name` вместо `id`;
   - `[pipeline]` содержит `stage_order` со списком stage name;
   - вместо `[frame_preprocessing]` использовать
     `[stages.input_preprocessing]` с `type = "frame_preprocessing"`;
   - визуализацию вынести из preset.
2. Сравнить с архитектурным примером в `llm_project_architecture.md`,
   раздел «Формат TOML preset v2».
3. Согласовать с пользователем перед тем, как мигрировать остальные пресеты.

В рамках этого `tasks.md` миграция не выполняется.

---

## Этап 3. План на восстановление публичного API `core/config/`

Сейчас `core/config/__init__.py` экспортирует только `RuntimeConfig`, а
потребители (`client/*`, `server/*`, `core/scenarios/*`, `core/nnet_interface/*`,
`core/stages/target_tracking/operations/yolo_target_tracker.py`) ещё
импортируют:

- `AVAILABLE_PRESETS`, `build_preset`, `get_preset_presentation`;
- `TrackerPreset`, `DEFAULT_PRESET_NAME`;
- `PROJECT_ROOT`, `load_app_config`;
- `NeuralConfig`, `VisualizationConfig`, `ClickSelectionConfig`;
- `PresetFieldReader` (через `core.config`).

Этап выполняется отдельной задачей и требует архитектурного решения:

1. Зафиксировать, какие из этих имён должны остаться публичными в новой
   архитектуре.
2. Перевести потребителей на новые пути (`core/preset/preset_field_reader.py`,
   `core/stages/...`).
3. Обновить `core/config/__init__.py` под актуальный публичный API.

Без согласования API не менять.

---

## Этап 4. Точечные исправления, требующие отдельного подтверждения

Каждое — отдельной задачей:

1. **`TargetRecoveryConfig.stage` default factory.** Сейчас
   `default_factory=lambda: StageConfig(enabled=True, operations=())` падает
   на дефолтной конструкции. Согласовать: оставить `enabled=False`, или сделать
   default factory возвращающим `StageConfig(enabled=False, operations=())`,
   или потребовать явный `stage`.
2. **`candidate_filtering` нейминг.** Config-классы используют `filter_type`,
   а архитектура — `operation_type`. Согласовать переименование с правкой
   импортов в `factory.py` / `manager.py` стадии.
3. **Чистка `_OLD.py` и `__OLD.py`.** После того, как все потребители
   `core.config` мигрированы, спланировать снос `preset_OLD.py`,
   `preset_loader_OLD.py`, `click_target_selector__OLD.py`. Не сейчас.

---

## Жёсткие ограничения, продолжающие действовать

- Не менять архитектуру preset v2.
- Не менять публичный API без отдельной фиксации в отчёте.
- Не удалять файлы.
- Не трогать `*_OLD.py` и `*__OLD.py`.
- Не добавлять зависимости.
- Не менять `poetry.lock`.
- Не выполнять Git-команды.
- Не делать массовые переименования.
- Не переносить визуализацию в core.
- Не возвращать `NeuralConfig` в root Preset.
- Не менять бизнес-логику tracking/selection/recovery без явной необходимости.

---

## Формат отчёта по этапу

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
