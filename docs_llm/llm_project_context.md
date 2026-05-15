# LLM Project Context: Thermal Tracker

## Назначение документа

Этот документ описывает только проектный контекст `thermal_tracker`.

Общие правила работы с пользователем и кодом находятся в:

```text
docs_llm/llm_global_rules.md
```

Перед выполнением задачи LLM должна читать документы в порядке:

1. `docs_llm/llm_global_rules.md`
2. `docs_llm/llm_project_context.md`
3. `docs_llm/llm_project_architecture.md`
4. `docs_llm/llm_project_handoff.md`
5. `docs_llm/llm_project_tasks.md`

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

Файлы с постфиксом `_OLD.py` являются временными архивными файлами. Их нельзя использовать как актуальную архитектуру.

---

## Текущий фокус работ

Сейчас идёт стабилизация новой stage/preset-архитектуры.

Основные задачи ближайшего этапа:

- синхронизировать импорты;
- проверить `core/preset/`;
- проверить `core/stages/config/`;
- проверить stage-level `config.py`, `manager.py`, `factory.py`;
- обновить preset TOML до формата v2;
- написать минимальные тесты на parsing;
- не менять архитектурные решения без согласования.
