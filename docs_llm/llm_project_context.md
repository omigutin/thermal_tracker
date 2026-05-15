# LLM Project Context: Thermal Tracker

## Назначение документа

Этот документ описывает только стабильный проектный контекст `thermal_tracker`.

Общие правила работы с пользователем и кодом находятся в:

```text
docs_llm/llm_global_rules.md
```

Подробная архитектура проекта находится в:

```text
docs_llm/llm_project_architecture.md
```

Текущий статус ремонта находится в:

```text
docs_llm/llm_project_handoff.md
```

---

## Проект

`thermal_tracker` — Python-проект для обработки тепловизионного видео, выделения целей и сопровождения выбранной цели.

Основные задачи проекта:

- подготовка кадров;
- стабилизация / оценка движения камеры;
- локализация движения;
- сборка кандидатов на цель;
- фильтрация ложных кандидатов;
- выбор цели;
- сопровождение цели;
- восстановление потерянной цели.

Проект связан с видео, Computer Vision / компьютерным зрением и потенциальной realtime-обработкой / обработкой в реальном времени.

---

## Основная архитектурная идея

Ядро проекта строится вокруг pipeline, состоящего из стадий.

Каждая стадия может иметь набор атомарных операций.

Общий поток:

```text
TOML preset
    ↓
PresetParser
    ↓
Preset
    ↓
StageConfigParser
    ↓
StageConfig[OperationConfig]
    ↓
Stage-specific Manager
    ↓
Stage-specific Factory
    ↓
Runtime operations
```

Runtime-код не должен работать с raw dict из TOML.

После парсинга весь pipeline должен получать строгие config-объекты.

---

## Важные директории

```text
src/thermal_tracker/core/preset/
src/thermal_tracker/core/stages/
src/thermal_tracker/core/stages/config/
src/thermal_tracker/core/config/
docs_llm/
presets/
tests/
```

Файлы с постфиксом `_OLD.py` или `__OLD.py` являются временными архивными файлами. Их нельзя использовать как актуальную архитектуру.

---

## Текущий фокус работ

Сейчас идёт стабилизация новой stage/preset-архитектуры.

Основные задачи ближайших этапов:

- синхронизировать импорты;
- проверить `core/preset/`;
- проверить `core/stages/config/`;
- проверить stage-level `config.py`, `manager.py`, `factory.py`;
- обновить preset TOML до формата v2;
- написать и прогнать минимальные тесты на parsing;
- не менять архитектурные решения без согласования или явного разрешения в текущем task-файле.

---

## Важное ограничение

Этот документ должен оставаться коротким.

Не добавлять сюда:

- подробную архитектуру;
- временный статус ремонта;
- результаты проверок;
- worklog;
- текущие задачи.

Для этого есть отдельные документы `llm_project_architecture.md`, `llm_project_handoff.md`, `llm_project_tasks.md`, `llm_worklog.md`.
