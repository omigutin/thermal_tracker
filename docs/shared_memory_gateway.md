# Shared Memory Gateway

Диагностический стенд имитирует будущую схему на Orange Pi:

```text
сетевой источник RAW Y8 -> gateway -> Shared Memory frames
                                   -> браузер
runtime worker <- Shared Memory commands
runtime worker -> Shared Memory results -> gateway -> браузер
```

## Формат кадра

- камера: `TWIN612G2 + MIPI Extender`;
- формат: `RAW Y8`;
- dtype: `uint8`;
- каналы: `1`;
- размер стенда по умолчанию: `512x640`, где OpenCV shape равен `(640, 512)`.

Gateway ставит локальный `timestamp_ns` в момент приёма кадра. Внешний timestamp отправителя сохраняется отдельно как `remote_timestamp_ns`, потому что часы разных устройств нельзя честно сравнивать без синхронизации.

## Полный локальный стенд

Синтетический источник без видеофайлов:

```powershell
poetry run python run_local_bench.py
```

С реальным видео:

```powershell
poetry run python run_local_bench.py --video-path "W:\path\to\video.mp4"
```

Открыть:

```text
http://127.0.0.1:8080
```

## Ручной запуск по процессам

Gateway:

```powershell
poetry run python run_gateway.py --host 0.0.0.0 --port 8080 --width 512 --height 640
```

Runtime worker:

```powershell
poetry run python run_shared_memory_worker.py --config configs/shared_memory_smoke.toml --report-every 50
```

Синтетический отправитель:

```powershell
poetry run python run_synthetic_sender.py --gateway-url http://127.0.0.1:8080 --fps 25 --width 512 --height 640
```

Отправитель видео:

```powershell
poetry run python run_network_sender.py "W:\path\to\video.mp4" --gateway-url http://127.0.0.1:8080 --fps 25 --width 512 --height 640 --loop
```

## Метрики

Главные поля в Web UI:

- `ingress_fps` — частота входящих кадров в gateway;
- `frame_id` — последний принятый кадр;
- `processed_frame` — последний обработанный кадр runtime;
- `frame_id_lag` — отставание runtime от входного потока;
- `processing_ms` — время обработки кадра;
- `ingress_to_runtime_ms` — задержка от приёма кадра gateway до старта обработки runtime;
- `source_to_result_ms` — задержка от приёма кадра gateway до готового результата.

Если `frame_id_lag` растёт, runtime не успевает. Если держится около `0-1`, поток обрабатывается нормально.

## Логи

Лог входящих кадров gateway:

```powershell
poetry run python run_gateway.py --ingress-log out\gateway_ingress.jsonl
```

Лог результатов runtime:

```powershell
poetry run python run_shared_memory_worker.py --metrics-log out\runtime_metrics.jsonl
```

В полном локальном стенде:

```powershell
poetry run python run_local_bench.py --ingress-log out\gateway_ingress.jsonl --metrics-log out\runtime_metrics.jsonl
```

## Очистка Shared Memory

Если процесс был остановлен аварийно:

```powershell
poetry run python run_shared_memory_cleanup.py
```
