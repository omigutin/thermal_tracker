# Thermal Tracker

Python-проект для сопровождения объектов на тепловизионном видео.

Код разделён на три пакета:

- `thermal_tracker_core` — алгоритмы, сценарии, конфиги, доменные модели, `connections` и `storage`;
- `thermal_tracker_server` — headless runtime, HTTP gateway и worker-процессы вокруг Shared Memory;
- `thermal_tracker_client` — desktop GUI, web-клиент и локальные отправители кадров.

Старый пакет `thermal_tracker` оставлен как compatibility-слой для старых импортов. Новый код лучше импортировать через `thermal_tracker_core`, `thermal_tracker_server` и `thermal_tracker_client`.

## Запуск

В корне оставлены только пользовательские входы:

```powershell
poetry run python run_server.py
poetry run python run_desktop_client.py
poetry run python run_web_client.py
```

Что они делают:

- `run_server.py` по умолчанию запускает серверный стек: HTTP gateway + runtime worker.
- `run_desktop_client.py` запускает desktop GUI для настройки и проверки пресетов.
- `run_web_client.py` открывает браузерный клиент уже запущенного gateway.

Технические режимы не исчезли, но убраны из корня:

```powershell
poetry run python run_server.py gateway
poetry run python run_server.py runtime
poetry run python run_server.py cleanup
poetry run python -m thermal_tracker_client.services.network_video_sender "W:\path\to\video.mp4"
poetry run python -m thermal_tracker_client.services.synthetic_network_sender
```

По умолчанию:

- `run_server.py` читает `configs/runtime.toml`;
- `run_desktop_client.py` читает `configs/dev.toml`;
- runtime/dev-конфиги лежат в `configs/`;
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

- `src/thermal_tracker_core/domain` — модели данных и контракты.
- `src/thermal_tracker_core/scenarios` — сборки режимов работы.
- `src/thermal_tracker_core/processing_stages` — алгоритмические стадии.
- `src/thermal_tracker_core/connections` — frames/commands/results и Shared Memory низкого уровня.
- `src/thermal_tracker_core/nnet_interface` — временный интерфейс к NN-моделям.
- `src/thermal_tracker_core/config` — загрузка TOML-конфигов и пресетов.
- `src/thermal_tracker_core/storage` — интерфейсы и заготовки хранилища истории.
- `src/thermal_tracker_server` — runtime, gateway и server-side процессы.
- `src/thermal_tracker_client/gui` — desktop GUI.
- `src/thermal_tracker_client/web` — статический Web UI gateway.
- `src/thermal_tracker_client/services` — локальные отправители и стенды.

## Документация

- [Architecture](docs/architecture.md)
- [Terminology](docs/terminology.md)
- [Project Structure](docs/project_structure.md)
- [Shared Memory Gateway](docs/shared_memory_gateway.md)
