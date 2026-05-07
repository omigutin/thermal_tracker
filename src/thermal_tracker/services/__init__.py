"""Совместимый слой для старых импортов `thermal_tracker.services.*`."""

from __future__ import annotations

import importlib
import sys


_SERVICE_ALIASES = {
    "gateway_service": "thermal_tracker.server.services.gateway_service",
    "shared_memory_cleanup": "thermal_tracker.server.services.shared_memory_cleanup",
    "shared_memory_runtime_worker": "thermal_tracker.server.services.shared_memory_runtime_worker",
    "local_bench": "thermal_tracker.client.services.local_bench",
    "network_video_sender": "thermal_tracker.client.services.network_video_sender",
    "shared_memory_video_simulator": "thermal_tracker.client.services.shared_memory_video_simulator",
    "synthetic_network_sender": "thermal_tracker.client.services.synthetic_network_sender",
}


for old_name, new_name in _SERVICE_ALIASES.items():
    module = importlib.import_module(new_name)
    sys.modules[f"{__name__}.{old_name}"] = module
    globals()[old_name] = module


__all__ = list(_SERVICE_ALIASES)
