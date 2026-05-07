# Architecture

Проект разделён на `core/server/client`, чтобы локальный стенд, будущий Orange Pi runtime и GUI-клиенты не мешали друг другу.

## Главная идея

GUI не является центром архитектуры. Центр — поток кадров, команды оператора, сценарии обработки и результаты. Один и тот же core должен работать в desktop GUI, в локальном gateway-стенде и в headless runtime на Orange Pi.

## Слои

- `thermal_tracker_core` содержит доменные модели, конфиги, сценарии, стадии обработки, адаптеры `connections` и `storage`.
- `thermal_tracker_server` содержит `runtime_app`, gateway и процессы, которые живут рядом с локальной Shared Memory.
- `thermal_tracker_client` содержит desktop GUI, web-ресурсы gateway и локальные отправители кадров для тестов.
- `thermal_tracker` оставлен только как compatibility-слой для старых импортов.

## Dev vs Runtime

Dev launch использует `configs/dev.toml`, `run_desktop_client.py` и desktop GUI. Это режим для визуальной настройки, проверки пресетов и записи диагностических видео.

Runtime/server launch использует `configs/runtime.toml` и `run_server.py`. Runtime по умолчанию ленивый: он собирает объекты, но не обязан сразу инициализировать сценарий с моделями.

Shared Memory низкого уровня находится в:

- `thermal_tracker_core/connections/frames/shared_memory_frame_reader.py`
- `thermal_tracker_core/connections/commands/shared_memory_command_reader.py`
- `thermal_tracker_core/connections/results/shared_memory_result_writer.py`
- `thermal_tracker_core/connections/shared_memory/`

## Gateway-Стенд

Gateway живёт в `thermal_tracker_server.services.gateway_service`: принимает RAW Y8 кадры по HTTP, пишет их в Shared Memory, читает результаты runtime и отдаёт их Web UI.

Web UI лежит отдельно в `thermal_tracker_client/web`. Это сознательное разделение: сервер обслуживает API и Shared Memory, клиентский интерфейс остаётся клиентским ресурсом.

## Scenario Selection

`ScenarioFactory` создаёт сценарии по имени из конфига:

- `nn_manual`
- `nn_auto`
- `opencv_manual`
- `opencv_auto_motion`

Legacy `kind` в алгоритмических пресетах оставлены для совместимости.
