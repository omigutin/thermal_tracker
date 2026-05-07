"""Писатель результатов в лог."""

from __future__ import annotations

import logging
from typing import Any

from .base_result_writer import BaseResultWriter


class LogResultWriter(BaseResultWriter):
    implementation_name = "log"
    is_ready = True

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def write(self, result: Any) -> None:
        self.logger.info("tracking_result=%r", result)
