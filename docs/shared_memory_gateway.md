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

## Основной запуск

Серверная сторона:

```powershell
poetry run python run_server.py --config configs/shared_memory_smoke.toml
```

Web-клиент:

```powershell
poetry run python run_web_client.py
```

Если браузер нужно открыть с другой машины, используйте адрес gateway:

```powershell
poetry run python run_web_client.py --server-url http://ORANGE_PI_IP:8080
```

## Отправка кадров для локального теста

Синтетический источник без видеофайлов:

```powershell
poetry run python -m thermal_tracker_client.services.synthetic_network_sender --gateway-url http://127.0.0.1:8080 --fps 25 --width 512 --height 640
```

Отправитель реального видео:

```powershell
poetry run python -m thermal_tracker_client.services.network_video_sender "W:\path\to\video.mp4" --gateway-url http://127.0.0.1:8080 --fps 25 --width 512 --height 640 --loop
```

Полный локальный bench с sender всё ещё есть, но это технический режим:

```powershell
poetry run python -m thermal_tracker_client.services.local_bench --video-path "W:\path\to\video.mp4"
```

## Ручные серверные режимы

Только gateway:

```powershell
poetry run python run_server.py gateway --host 0.0.0.0 --port 8080 --width 512 --height 640
```

Только runtime worker:

```powershell
poetry run python run_server.py runtime --config configs/shared_memory_smoke.toml --report-every 50
```

Cleanup Shared Memory:

```powershell
poetry run python run_server.py cleanup
```

Cleanup не запускается автоматически. Если включить его перед стартом сервера, это нужно делать явно:

```powershell
poetry run python run_server.py --cleanup-before-start
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
poetry run python run_server.py gateway --ingress-log out\gateway_ingress.jsonl
```

Лог результатов runtime:

```powershell
poetry run python run_server.py runtime --metrics-log out\runtime_metrics.jsonl
```

В режиме `all`:

```powershell
poetry run python run_server.py --ingress-log out\gateway_ingress.jsonl --metrics-log out\runtime_metrics.jsonl
```
