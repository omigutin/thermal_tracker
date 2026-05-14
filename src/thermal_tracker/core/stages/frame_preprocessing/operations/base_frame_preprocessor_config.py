from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaseFramePreprocessorConfig:
    """Базовые настройки операции предобработки кадра."""

    # Включает или отключает операцию.
    enabled: bool = True

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

    @staticmethod
    def _validate_non_negative_int(value: int, field_name: str) -> None:
        """Проверить, что целое значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")

    @staticmethod
    def _validate_positive_float(value: float, field_name: str) -> None:
        """Проверить, что вещественное значение положительное."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")

    @staticmethod
    def _validate_non_negative_float(value: float, field_name: str) -> None:
        """Проверить, что вещественное значение неотрицательное."""
        if value < 0:
            raise ValueError(f"{field_name} must be greater than or equal to 0.")

    @staticmethod
    def _validate_ratio(value: float, field_name: str) -> None:
        """Проверить, что значение находится в диапазоне [0, 1]."""
        if not 0 <= value <= 1:
            raise ValueError(f"{field_name} must be in range [0, 1].")

    @staticmethod
    def _validate_percent(value: float, field_name: str) -> None:
        """Проверить, что значение является процентом."""
        if not 0 <= value <= 100:
            raise ValueError(f"{field_name} must be in range [0, 100].")

    @staticmethod
    def _validate_odd_positive_kernel(value: int, field_name: str) -> None:
        """Проверить, что размер ядра положительный и нечётный."""
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd.")

    @staticmethod
    def _validate_sobel_like_kernel(value: int, field_name: str) -> None:
        """Проверить размер ядра OpenCV-фильтра типа Sobel/Laplacian."""
        if value not in (1, 3, 5, 7):
            raise ValueError(f"{field_name} must be one of: 1, 3, 5, 7.")

    @staticmethod
    def _normalize_odd_kernel(value: int) -> int:
        """Вернуть ближайший нечётный размер ядра не меньше 1."""
        normalized = max(1, int(value))
        return normalized if normalized % 2 == 1 else normalized + 1
