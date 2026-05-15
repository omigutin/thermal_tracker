# LLM Index: Thermal Tracker

## Purpose

This is the main entry point for any LLM working with the project.

Read this file first, then load the specific documents required for the current task.

---

## Documents

| Document | Purpose | When to read |
|---|---|---|
| `docs_llm/llm_global_rules.md` | Общие правила работы с пользователем и кодом | Всегда перед задачей |
| `docs_llm/llm_project_context.md` | Краткий контекст проекта `thermal_tracker` | Всегда перед проектной задачей |
| `docs_llm/llm_project_architecture.md` | Архитектура stages, presets, parsing, managers, factories | Перед изменением архитектуры, импортов, конфигов, stages |
| `docs_llm/llm_project_handoff.md` | Текущее состояние рефакторинга и известные риски | Перед продолжением текущего этапа |
| `docs_llm/llm_project_tasks.md` | Текущая конкретная задача для LLM | Всегда перед выполнением |

---

## Required reading order

1. `docs_llm/llm_global_rules.md`
2. `docs_llm/llm_project_context.md`
3. `docs_llm/llm_project_architecture.md`
4. `docs_llm/llm_project_handoff.md`
5. `docs_llm/llm_project_tasks.md`

---

## Обязательная фиксация после каждого этапа

После завершения каждого существенного этапа LLM обязана обновить LLM-документы.

Существенными этапами считаются:

1. аудит без изменений;
2. исправление импортов;
3. стабилизация `core/preset`;
4. стабилизация `core/stages/config`;
5. добавление или исправление тестов;
6. запуск проверок;
7. исправление ошибок после проверок;
8. обновление обычной документации;
9. подготовка итогового отчёта.

После каждого этапа обновить:

- `docs_llm/llm_project_handoff.md` — что сделано, что найдено, что осталось, какие риски;
- `docs_llm/llm_project_tasks.md` — текущий статус задачи и следующий рекомендуемый шаг.

Если в ходе работы принято стабильное архитектурное решение, обновить также:

- `docs_llm/llm_project_architecture.md`.

Если найдено новое общее правило работы с кодом или LLM, не добавлять его автоматически в глобальные правила. 
Сначала вынести в отчёт как предложение.

---

## Document update commands

When the user says one of the commands below, update the corresponding document.

### “Зафиксируй это в моих правилах написания кода”

Update:

`docs_llm/llm_global_rules.md`

Meaning:

- add or refine a general coding/workflow rule;
- place it into the correct section;
- avoid duplicating existing rules;
- if the rule conflicts with an existing rule, report the conflict before editing.

### “Обнови глобальные правила LLM”

Update:

`docs_llm/llm_global_rules.md`

Meaning:

- update general rules that apply across projects;
- do not add project-specific architecture here.

### “Обнови контекст проекта”

Update:

`docs_llm/llm_project_context.md`

Meaning:

- update stable project facts;
- keep it short;
- do not put detailed architecture here.

### “Обнови архитектуру проекта”

Update:

`docs_llm/llm_project_architecture.md`

Meaning:

- update stage/preset architecture;
- document accepted architectural decisions;
- do not include temporary task status here.

### “Обнови handoff”

Update:

`docs_llm/llm_project_handoff.md`

Meaning:

- update current refactoring status;
- add known issues, risks, completed work, next safe steps;
- keep it useful for another LLM continuing the work.

### “Обнови задачу для LLM”

Update:

`docs_llm/llm_project_tasks.md`

Meaning:

- rewrite the current task;
- include goal, scope, restrictions, expected report format, checks;
- this file may be fully overwritten each stage.

### “Сформируй задачу для другой LLM”

Update:

`docs_llm/llm_project_tasks.md`

Optional update:

`docs_llm/llm_project_handoff.md`

Meaning:

- create a precise executable task;
- include strict restrictions;
- require audit/plan before changes unless user explicitly allows implementation.

### “Зафиксируй архитектурное решение”

Update:

`docs_llm/llm_project_architecture.md`

Optional update:

`docs_llm/llm_project_handoff.md`

Meaning:

- add the decision to architecture if it is stable;
- add to handoff if it affects current refactoring.

### “Зафиксируй текущее состояние”

Update:

`docs_llm/llm_project_handoff.md`

Meaning:

- record what is done, what is broken, what remains, what not to touch.

---

## Rules for updating LLM documents

- Do not scatter the same rule across multiple files.
- `llm_global_rules.md` is for cross-project rules.
- `llm_project_context.md` is for short stable project context.
- `llm_project_architecture.md` is for stable architecture.
- `llm_project_handoff.md` is for current transition state.
- `llm_project_tasks.md` is for the current executable task.
- If unsure where a rule belongs, ask the user before editing.
