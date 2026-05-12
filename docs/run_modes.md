# Режимы запуска

Документ описывает, какие процессы запускаются и зачем.

## `run_server.py`

Основной серверный вход. Поддерживает режимы:

- `all` — запускает и gateway, и runtime worker;
- `gateway` — только HTTP gateway;
- `runtime` — только обработчик кадров;
- `cleanup` — удаляет старые shared memory сегменты.

По умолчанию используется конфиг `configs/server.toml`.

## `run_desktop_client.py`

Запускает desktop GUI. Обычно используется для:

- ручного клика по цели;
- проверки и настройки пресетов;
- визуальной диагностики.

По умолчанию использует `configs/desktop_client.toml`.

## `run_web_client.py`

Открывает web-интерфейс для уже работающего gateway.

По умолчанию использует `configs/web_client.toml`.
