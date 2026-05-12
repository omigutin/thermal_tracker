# Структура проекта

Ниже показана актуальная структура верхнего уровня.

```text
src/
  thermal_tracker/
    core/
      config/
      connections/
      domain/
      nnet_interface/
      processing_stages/
      scenarios/
      state_machine/
      storage/
    server/
      cli.py
      runtime_app.py
      services/
    client/
      cli.py
      gui/
      services/

run_server.py
run_desktop_client.py
run_web_client.py
configs/
presets/
models/
trackers/
docs/
tests/
```

## Смысл модулей

- `core` — алгоритмическое ядро.
- `server` — серверные процессы и gateway.
- `client` — пользовательские интерфейсы и локальные сервисы.
