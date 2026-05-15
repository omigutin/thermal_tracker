"""Совместимый слой для старого имени пакета `thermal_tracker`.

Новый код разнесён по пакетам:
- `thermal_tracker.core` — алгоритмы, конфиги, доменные модели и адаптеры данных;
- `thermal_tracker.server` — runtime, gateway и серверные процессы;
- `thermal_tracker.client` — desktop GUI, web-ресурсы и локальные отправители.
"""

from __future__ import annotations

import importlib
import sys


_ALIASES = {
    "config": "thermal_tracker.core.config",
    "connections": "thermal_tracker.core.connections",
    "domain": "thermal_tracker.core.domain",
    "errors": "thermal_tracker.core.errors",
    "logging": "thermal_tracker.core.logging",
    "nnet_interface": "thermal_tracker.core.nnet_interface",
    # `processing_stages` сознательно удалён: модуля
    # `thermal_tracker.core.processing_stages` нет, новый путь -
    # `thermal_tracker.core.stages`. Алиас под именем `stages` не вводится,
    # чтобы не расширять публичный API без согласования.
    "scenarios": "thermal_tracker.core.scenarios",
    "storage": "thermal_tracker.core.storage",
    "runtime_app": "thermal_tracker.server.runtime_app",
    "cli": "thermal_tracker.client.cli",
    "gui": "thermal_tracker.client.gui",
    "session": "thermal_tracker.client.session",
}


for old_name, new_name in _ALIASES.items():
    module = importlib.import_module(new_name)
    sys.modules[f"{__name__}.{old_name}"] = module
    globals()[old_name] = module


def main() -> None:
    """Запускает desktop GUI через новый клиентский пакет."""

    from thermal_tracker.client.gui.app import run_gui as _main

    _main()


__all__ = ["main"]
