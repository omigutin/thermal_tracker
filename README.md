# Thermal Tracker

Thermal Tracker — это проект для сопровождения объектов на тепловизионном видео.

Если говорить просто: система получает видеокадры, ищет на них нужный объект, удерживает его в фокусе и сообщает результат оператору.

## С чего начать

1. Прочитайте [Быстрый старт](docs/quick_start.md).
2. Посмотрите [Режимы запуска](docs/run_modes.md).
3. Для настройки качества перейдите в [Пресеты](docs/presets.md).
4. Чтобы понять внутреннюю логику, откройте [Стадии обработки](docs/processing_stages.md).

## Основные документы

- [Быстрый старт](docs/quick_start.md)
- [Режимы запуска](docs/run_modes.md)
- [Конфигурация](docs/configuration.md)
- [Пресеты](docs/presets.md)
- [Стадии обработки](docs/processing_stages.md)
- [Сценарии](docs/scenarios.md)
- [IRST-трекинг](docs/irst_tracking.md)
- [Архитектура](docs/architecture.md)
- [Структура проекта](docs/project_structure.md)
- [Термины](docs/terminology.md)
- [Диагностика проблем](docs/troubleshooting.md)

## Коротко о составе проекта

- `src/thermal_tracker/core` — логика обработки кадров, трекинга и восстановления цели.
- `src/thermal_tracker/server` — серверный запуск, gateway и runtime worker.
- `src/thermal_tracker/client` — desktop-интерфейс и web-клиент.

## Лицензия

MIT.
