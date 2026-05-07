# Project Structure

```text
src/
  thermal_tracker_core/
    config/
    connections/
      frames/
      commands/
      results/
      shared_memory/
    domain/
    nnet_interface/
    processing_stages/
    scenarios/
    storage/

  thermal_tracker_server/
    cli.py
    runtime_app.py
    services/
      gateway_service.py
      shared_memory_runtime_worker.py
      shared_memory_cleanup.py

  thermal_tracker_client/
    cli.py
    web_client.py
    gui/
    web/
      index.html
      app.js
      styles.css
    services/
      local_bench.py
      network_video_sender.py
      shared_memory_video_simulator.py
      synthetic_network_sender.py

  thermal_tracker/
    __init__.py
    services/
      __init__.py

run_server.py
run_desktop_client.py
run_web_client.py
models/
trackers/
presets/
configs/
video/
docs/
out/
```

## Root Entrypoints

- `run_server.py` — основной запуск серверной стороны. Режимы: `all`, `gateway`, `runtime`, `cleanup`.
- `run_desktop_client.py` — desktop GUI.
- `run_web_client.py` — открытие Web UI запущенного gateway.

Стендовые отправители запускаются как модули, а не отдельными файлами в корне.

## Processing Stages

- `frame_preprocessing` подготавливает кадры.
- `frame_stabilization` оценивает движение камеры/кадра.
- `moving_area_detection` ищет движущиеся области.
- `target_candidate_extraction` выделяет кандидатов на цель.
- `candidate_filtering` фильтрует ложные кандидаты.
- `target_selection` выбирает цель по клику или кандидатам.
- `target_tracking` сопровождает выбранную цель.
- `target_recovery` отвечает за повторный поиск после потери.

## Connections

- `connections/frames` читает кадры из видео, последовательности изображений, RTSP, replay или Shared Memory.
- `connections/commands` читает команды оператора/runtime из GUI, Shared Memory или null-reader.
- `connections/results` пишет результаты в GUI, Shared Memory, файлы или лог.

Launch-конфиги живут в `configs/dev.toml` и `configs/runtime.toml`. Алгоритмические пресеты остаются в `presets/`.
