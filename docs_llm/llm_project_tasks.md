# LLM Project Tasks: Resume autonomous stabilization after interrupted run

## Назначение документа

Это текущая исполняемая задача для LLM.

Файл можно полностью перезаписывать после завершения этапа.

---

## Обязательное чтение перед работой

Перед выполнением задачи прочитать:

1. `docs_llm/llm_index.md`
2. `docs_llm/llm_global_rules.md`
3. `docs_llm/llm_project_context.md`
4. `docs_llm/llm_project_architecture.md`
5. `docs_llm/llm_project_handoff.md`
6. `docs_llm/llm_project_tasks.md`
7. `docs_llm/llm_worklog.md`, если файл существует

---

## Режим работы

Текущая задача разрешает ограниченный автономный режим.

LLM может выполнять действия из этого файла без дополнительного подтверждения пользователя.

Если требуется действие вне scope, LLM должна остановиться, записать вопрос в отчёт и не выполнять спорное изменение.

---

## Разрешённые локальные Git-команды

В рамках текущей задачи пользователь разрешает локальные коммиты после каждого завершённого этапа.

Разрешено:

```bash
git status
git diff
git log --oneline -n 10
git add <changed_files>
git commit -m "<message>"
```

Запрещено:

```bash
git push
git pull
git reset
git rebase
git checkout
git clean
git stash
```

Правило:

```text
один завершённый этап = один локальный коммит
```

Формат commit message:

```text
llm: <stage> - <short result>
```

Примеры:

```text
llm: docs - fix autonomous workflow instructions
llm: tests - verify preset parser checks
llm: imports - stabilize stage config imports
```

---

## Никогда не добавлять в коммит

Запрещено добавлять:

```text
outputs/
sandbox/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
.env
*.pyc
*.pyo
*.log
artifacts/
videos/
datasets/
weights/
```

Перед каждым коммитом выполнить `git status` и проверить список файлов.

---

## Цель текущего этапа

Продолжить работу после остановки предыдущей LLM.

Нужно:

1. проверить, какие изменения уже сделаны;
2. привести LLM-документы в согласованное состояние;
3. закоммитить уже выполненный этап, если он ещё не закоммичен;
4. очистить cache-файлы;
5. запустить проверки;
6. исправить только ошибки в рамках текущей архитектуры;
7. обновить LLM-документы;
8. сделать локальные коммиты по этапам.

---

## Scope

Проверять и исправлять можно:

```text
docs_llm/
CLAUDE.md
src/thermal_tracker/core/preset/
src/thermal_tracker/core/stages/config/
src/thermal_tracker/core/stages/target_recovery/
src/thermal_tracker/__init__.py
tests/thermal_tracker/core/preset/
tests/thermal_tracker/core/stages/config/
tests/thermal_tracker/core/stages/target_recovery/
```

Не выходить за этот scope без явной причины, связанной с ошибкой проверок.

---

## Этап 0. Meta-fix LLM-документов

Сначала проверить и, если нужно, исправить LLM-документы.

Обязательные проверки:

1. Главный индекс должен называться:

```text
docs_llm/llm_index.md
```

Не должно быть основного файла:

```text
docs_llm/llm_index.md.md
```

2. `llm_index.md` должен содержать `llm_worklog.md` в таблице документов и порядке чтения.

3. Во всех документах актуальный путь к `PresetFieldReader`:

```text
src/thermal_tracker/core/preset/field_reader.py
```

Класс:

```text
PresetFieldReader
```

Неверный путь:

```text
src/thermal_tracker/core/preset/preset_field_reader.py
```

4. `llm_global_rules.md` должен содержать раздел “Автономный режим”.

5. `llm_project_tasks.md` должен явно разрешать локальные Git-коммиты только для текущего этапа.

После исправления документов:

- обновить `llm_project_handoff.md`;
- добавить запись в `llm_worklog.md`;
- сделать локальный коммит:

```bash
git commit -m "llm: docs - fix autonomous workflow instructions"
```

---

## Этап 1. Проверить состояние Git и предыдущих изменений

Выполнить:

```bash
git status
git log --oneline -n 10
```

Если есть незакоммиченные изменения предыдущей LLM:

1. проверить, что они входят в scope;
2. проверить, что нет запрещённых файлов;
3. если изменения допустимы — сделать локальный коммит:

```bash
git commit -m "llm: checkpoint - previous stabilization changes"
```

Если изменения спорные — не коммитить их, зафиксировать в отчёте и остановиться.

---

## Этап 2. Очистить Python cache

Удалить только cache-директории и compiled-файлы:

```text
__pycache__/
*.pyc
*.pyo
.pytest_cache/
```

Не использовать `git clean`.

Предпочтительно удалить средствами Python или PowerShell, если доступно.

Не удалять исходные файлы.

После очистки — обновить worklog.

---

## Этап 3. Запустить compileall

Выполнить:

```bash
python -m compileall src/thermal_tracker/core
```

Если используется Windows PowerShell, команда та же.

Если команда падает:

1. зафиксировать полный вывод;
2. исправлять только ошибки импортов, синтаксиса или очевидных несоответствий новой архитектуре;
3. не менять бизнес-логику tracking/selection/recovery;
4. после исправления повторить проверку.

После успешного этапа:

- обновить `llm_project_handoff.md`;
- обновить `llm_project_tasks.md`;
- добавить запись в `llm_worklog.md`;
- сделать локальный коммит:

```bash
git commit -m "llm: checks - pass core compileall"
```

---

## Этап 4. Запустить pytest по добавленным тестам

Выполнить:

```bash
python -m pytest tests/thermal_tracker/core/preset tests/thermal_tracker/core/stages/config tests/thermal_tracker/core/stages/target_recovery -v
```

Если pytest недоступен:

- не устанавливать новые зависимости;
- зафиксировать это в отчёте;
- не создавать самодельный pytest shim в проекте;
- можно выполнить только синтаксическую проверку тестов через compileall.

Если тесты падают:

1. исправлять только код/тесты в scope;
2. не менять архитектуру;
3. не менять публичный API без фиксации и явного разрешения текущей задачи;
4. повторить pytest.

После успешного этапа:

- обновить `llm_project_handoff.md`;
- обновить `llm_project_tasks.md`;
- добавить запись в `llm_worklog.md`;
- сделать локальный коммит:

```bash
git commit -m "llm: tests - pass preset and stage config tests"
```

---

## Этап 5. Подготовить следующий task-файл

После проверок обновить `docs_llm/llm_project_tasks.md` под следующий этап.

Если compileall и pytest прошли, следующий рекомендуемый этап:

```text
Plan TOML preset v2 migration
```

Если проверки не прошли, следующий task должен быть:

```text
Fix remaining compileall/pytest failures
```

Не начинать миграцию TOML-пресетов автоматически в рамках этой задачи.

---

## Жёсткие ограничения

- Не менять архитектуру preset v2.
- Не менять публичный API без фиксации в отчёте.
- Не удалять исходные файлы.
- Не трогать `*_OLD.py` и `*__OLD.py`.
- Не добавлять зависимости.
- Не менять `poetry.lock`.
- Не выполнять `git push`.
- Не делать массовые переименования.
- Не переносить визуализацию в core.
- Не возвращать `NeuralConfig` в root Preset.
- Не менять бизнес-логику tracking/selection/recovery без явной необходимости.
- Не коммитить временные каталоги и cache-файлы.

---

## Итоговый отчёт

После завершения задачи вернуть:

```text
Итог:
- Что изменено:
- Почему так:
- Затронутые файлы:
- Локальные коммиты:
- Проверки выполнены:
- Проверки не выполнены:
- Остаточные риски:
- Следующий рекомендуемый шаг:
```
