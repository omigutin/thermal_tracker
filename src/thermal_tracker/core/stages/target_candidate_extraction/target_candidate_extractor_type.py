"""Типы методов сборки кандидатов на цель."""

from __future__ import annotations

from enum import StrEnum


class TargetCandidateExtractorType(StrEnum):
    """Доступные методы стадии target_candidate_extraction."""

    OPENCV_CONNECTED_COMPONENTS = "opencv_connected_components"  # Собирает кандидатов из компонент связности маски.
    OPENCV_CONTOUR = "opencv_contour"  # Собирает кандидатов по внешним контурам маски.

    # BLOB = "blob"  # Поиск компактных пятен как отдельных кандидатов.
    # DISTANCE_TRANSFORM = "distance_transform"  # Разделение слипшихся объектов через distance transform.
    # WATERSHED = "watershed"  # Разделение слипшихся объектов водоразделом.
