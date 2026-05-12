# Shared Memory Gateway

Диагностический стенд имитирует схему запуска, где gateway принимает кадры,
пишет их в shared memory, а runtime worker читает кадры и возвращает результаты.

```text
сетевой источник RAW Y8 -> gateway -> Shared Memory frames
                                  -> браузер
runtime worker <- Shared Memory commands
runtime worker -> Shared Memory results -> gateway -> браузер
```

## Формат кадра

- формат: `RAW Y8`;
- dtype: `uint8`;
- каналы: `1`;
- размер стенда по умолчанию: `512x640`, где OpenCV shape равен `(640, 512)`.

Gateway ставит локальный `timestamp_ns` в момент приёма кадра.

## Основной запуск

Серверная сторона:

```bash
poetry run python run_server.py --config configs/server.toml
```

Web-клиент:

```bash
poetry run python run_web_client.py
```

## Отправка кадров для локального теста

Синтетический источник:

```bash
poetry run python -m thermal_tracker.client.services.synthetic_network_sender --gateway-url http://127.0.0.1:8080 --fps 25 --width 512 --height 640
```

Отправитель видео:

```bash
poetry run python -m thermal_tracker.client.services.network_video_sender "path/to/video.mp4" --gateway-url http://127.0.0.1:8080 --fps 25 --width 512 --height 640 --loop
```

Локальный bench:

```bash
poetry run python -m thermal_tracker.client.services.local_bench --video-path "path/to/video.mp4"
```

## Ручные серверные режимы

Только gateway:

```bash
poetry run python run_server.py gateway --host 0.0.0.0 --port 8080 --width 512 --height 640
```

Только runtime worker:

```bash
poetry run python run_server.py runtime --config configs/server.toml --report-every 50
```

Очистка shared memory:

```bash
poetry run python run_server.py cleanup
```
