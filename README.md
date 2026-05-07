# Thermal Tracker

Python-проект для сопровождения объектов на тепловизионном видео.

Код организован вокруг пакета `thermal_tracker`: сценарии отдельно, стадии обработки отдельно, GUI отдельно, внешние подключения отдельно. GUI не является центром архитектуры, а остаётся dev-стендом для проверки и настройки.

## Запуск

```bash
python run_dev.py
python run_runtime.py
```

По умолчанию:

- `run_dev.py` читает `configs/dev.toml`;
- `run_runtime.py` читает `configs/runtime.toml`;
- алгоритмические пресеты лежат в `presets/`;
- модели лежат в `models/`;
- конфиги внешних трекеров лежат в `trackers/`.

## Пресеты

Основные имена:

- `opencv_general`
- `opencv_small_target`
- `opencv_clutter`
- `yolo_general`
- `yolo_auto`

## Структура

- `src/thermal_tracker/domain` — модели данных и контракты.
- `src/thermal_tracker/scenarios` — сборки режимов работы.
- `src/thermal_tracker/processing_stages` — алгоритмические стадии.
- `src/thermal_tracker/connections` — frames/commands/results.
- `src/thermal_tracker/nnet_interface` — временный интерфейс к NN-моделям.
- `src/thermal_tracker/gui` — dev GUI.
- `src/thermal_tracker/config` — загрузка TOML-конфигов и пресетов.
- `src/thermal_tracker/storage` — интерфейсы и заготовки хранилища истории.

## Документация

- [Architecture](docs/architecture.md)
- [Terminology](docs/terminology.md)
- [Project Structure](docs/project_structure.md)
- [Shared Memory Gateway](docs/shared_memory_gateway.md)
