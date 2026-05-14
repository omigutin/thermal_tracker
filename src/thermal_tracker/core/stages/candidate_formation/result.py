from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...domain.models import BoundingBox


@dataclass(slots=True)
class CandidateFormerResult:
    """Кандидат, сформированный из маски движения или другого источника."""

    bbox: BoundingBox
    area: int
    confidence: float = 0.0
    label: str = "candidate"
    mask: Optional[np.ndarray] = None
    class_id: int | None = None

    def __post_init__(self) -> None:
        """Проверить базовую согласованность кандидата."""
        if self.area < 0:
            raise ValueError("area must be greater than or equal to 0.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0.0, 1.0].")

        if self.class_id is not None and self.class_id < 0:
            raise ValueError("class_id must be greater than or equal to 0.")
