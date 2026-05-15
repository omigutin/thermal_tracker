# LLM Project Handoff: Thermal Tracker refactoring

## Назначение документа

Этот файл фиксирует текущее состояние рефакторинга, чтобы LLM могла безопасно
продолжить рутинные задачи: аудит импортов, тесты, документацию и простые
исправления.

Перед работой обязательно прочитать:

1. `docs_llm/llm_global_rules.md`
2. `docs_llm/llm_project_context.md`
3. `docs_llm/llm_project_architecture.md`
4. `docs_llm/llm_project_handoff.md`
5. `docs_llm/llm_project_tasks.md`
6. `docs_llm/llm_worklog.md` — журнал этапов; читать перед началом нового этапа.

---

## Текущий статус (на 2026-05-15)

Прошёл этап «аудит + минимальные импортные правки + тесты на парсинг».

Сделано:

- Прочитан весь scope: `core/preset/`, `core/stages/`, `core/stages/config/`,
  `core/config/`, корневой `thermal_tracker/__init__.py`, репрезентативный TOML
  пресет, `tests/`.
- В `core/preset/parser.py` импорт `TargetRecoveryConfig` переведён с
  `..config.preset_OLD` на `..stages.target_recovery.config`. Поведение
  парсера не менялось, публичный API парсера не менялся.
- В корневом `src/thermal_tracker/__init__.py` удалён битый alias
  `"processing_stages": "thermal_tracker.core.processing_stages"`. Новый alias
  `stages` не добавлен, чтобы не расширять публичный API без согласования.
- Добавлены изолированные unit-тесты на:
  - `PresetFieldReader`;
  - `StageConfig`;
  - `StageConfigParser`;
  - `TargetRecoveryConfig`.
- Логика этих тестов прошла 34/34 в sandbox-копии (compileall + собственный
  раннер).

Запустить полноценный compileall и pytest на реальном дереве из текущей
Linux-песочницы не удалось — этому мешают артефакты virtiofs-монтирования
Windows-каталога и устаревший `__pycache__`. Подробности в
`docs_llm/llm_worklog.md`.

---

## Уже принятые архитектурные решения

- Stage-директории не нумеровать.
- Порядок выполнения задаётся через `pipeline.stage_order`.
- Новый preset package: `src/thermal_tracker/core/preset/`.
- `PresetFieldReader` остаётся в `core/preset/preset_field_reader.py`
  (имя файла `preset_field_reader.py`, не `field_reader.py`).
- Не переименовывать `PresetFieldReader` в `ConfigFieldReader`.
- Общий `StageConfig` живёт в `core/stages/config/`.
- Использовать TOML preset v2 через `[stages.<stage_name>]`.
- Не добавлять `VisualizationConfig` в root `Preset`.
- Не добавлять отдельный `NeuralConfig` в root `Preset`.
- `StagePreset.name`, не `StagePreset.id`.
- Файлы `_OLD.py` игнорировать. Это касается также `click_target_selector__OLD.py`
  (двойное подчёркивание перед `OLD` — фактически тот же случай).

---

## Известные проблемы вне разрешённого scope текущего шага

Список найденных, но **намеренно** не правленых проблем. Снять с них
ограничения можно только отдельной задачей.

1. **Старый публичный API `core/config/` сохраняется только частично.**
   `core/config/__init__.py` экспортирует только `RuntimeConfig`. Но во многих
   потребителях ещё импортируются `AVAILABLE_PRESETS`, `build_preset`,
   `get_preset_presentation`, `TrackerPreset`, `DEFAULT_PRESET_NAME`,
   `PROJECT_ROOT`, `load_app_config`, `NeuralConfig`, `VisualizationConfig`,
   `ClickSelectionConfig`, `PresetFieldReader`. Эти потребители битые:
   `client/cli.py`, `client/session.py`, `client/web_client.py`,
   `client/gui/app.py`, `client/gui/frame_visualizer.py`,
   `client/gui/video_workspace_window.py`, `core/scenarios/*.py`,
   `core/nnet_interface/yolo_nnet_interface.py`,
   `core/stages/target_tracking/operations/yolo_target_tracker.py`,
   `server/cli.py`, `server/runtime_app.py`, `server/services/gateway_service.py`,
   `server/services/web_recording.py`.
2. **TOML-пресеты в старом формате.** Все `presets/*.toml` (`opencv_general`,
   `opencv_clutter`, `opencv_small_target`, `opencv_auto_motion`,
   `irst_small_target`, `yolo_auto`, `yolo_general`) используют:
   - `[meta]` с `id` вместо `name`;
   - секции стадий `[frame_preprocessing]` вместо `[stages.<name>]`;
   - отсутствует `pipeline.stage_order`;
   - присутствует `[visualization]` в корне.
   `PresetParser` не разберёт эти файлы.
3. **Латентный баг в `TargetRecoveryConfig.stage` default factory.** Вызов
   `TargetRecoveryConfig()` без аргументов падает: `default_factory` создаёт
   `StageConfig(enabled=True, operations=())`, что нарушает инвариант
   `StageConfig.__post_init__`. На текущий парсинг не влияет (парсер всегда
   передаёт явный `stage`), но проявится при любом дефолтном конструировании.
4. **Расхождение нейминга в `candidate_filtering`.** Config-классы используют
   `filter_type: ClassVar[CandidateFilterType]`, а архитектура предписывает
   `operation_type: ClassVar[Enum]`.
5. **Stale `__pycache__`.** В `src/thermal_tracker/core/config/__pycache__`
   лежат `.pyc` для `app_config.py` и `preset.py`, которых на диске нет.
   Из Linux-песочницы их удалить не получилось (`Operation not permitted`).
   На Windows-стороне обязательно снести `__pycache__` перед запуском Python.

---

## Что нельзя делать без отдельного согласования

- Менять архитектуру preset v2.
- Нумеровать директории стадий.
- Переносить визуализацию в core preset.
- Возвращать `NeuralConfig` в root `Preset`.
- Менять публичный API managers/factories без плана.
- Удалять `_OLD.py` и `__OLD.py` файлы.
- Удалять `preset_OLD.py` и `preset_loader_OLD.py` без поиска импортов.
- Добавлять новые зависимости.
- Менять `poetry.lock`.
- Запускать Git-команды.
- Делать массовые переименования без согласования.

---

## Рекомендуемый следующий порядок

1. Пользователю — снести `__pycache__` целиком и прогнать на Windows-стороне:
   - `python -m compileall src/thermal_tracker/core`;
   - `python -m pytest tests/thermal_tracker/core -v`.
2. Отдельная задача: миграция TOML-пресетов в формат v2.
3. Отдельная задача: ревизия публичного API `core/config/` и зачистка битых
   импортов в `client/`, `server/`, `core/scenarios/`, `core/nnet_interface/`,
   `core/stages/target_tracking/operations/yolo_target_tracker.py`.
4. Отдельная задача: починка default-конфига `TargetRecoveryConfig.stage`.
5. Отдельная задача: решение по нейминговому расхождению
   `filter_type` vs `operation_type` в `candidate_filtering`.
