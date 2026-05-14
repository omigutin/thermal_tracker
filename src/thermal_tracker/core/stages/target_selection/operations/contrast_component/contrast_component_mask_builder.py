from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ...result import TargetPolarity


class ContrastComponentMaskBuilderSettings(Protocol):
    """Описывает настройки, нужные построителю масок по клику."""

    # Радиус локального окна вокруг клика для оценки шума и разброса яркости.
    local_window_radius: int
    # Коэффициент, который превращает локальное стандартное отклонение в допуск яркости.
    similarity_sigma: float
    # Минимальный допустимый порог похожести по яркости.
    min_tolerance: int
    # Максимальный допустимый порог похожести по яркости.
    max_tolerance: int

    # Размер ядра морфологической очистки бинарной маски.
    mask_morphology_kernel: int
    # Количество итераций удаления мелкого шума из маски.
    mask_open_iterations: int
    # Количество итераций заполнения небольших разрывов внутри маски.
    mask_close_iterations: int

    # Размер ядра размытия для резервной score-маски.
    score_blur_kernel: int
    # Размер ядра оператора Sobel для оценки локальных границ.
    score_sobel_kernel: int
    # Вес похожести пикселя на кликнутую яркость в резервной score-маске.
    score_difference_weight: float
    # Вес локального градиента в резервной score-маске.
    score_gradient_weight: float
    # Вес расстояния от клика в резервной score-маске.
    score_distance_weight: float
    # Максимальное значение score, при котором пиксель попадает в резервную маску.
    score_threshold: float


@dataclass(slots=True)
class ContrastComponentMaskBuilder:
    """Строит маски выбранной цели вокруг точки клика."""

    config: ContrastComponentMaskBuilderSettings

    def build_mask(self, patch: np.ndarray, click_x: int, click_y: int) -> tuple[np.ndarray, TargetPolarity]:
        """
            Построить основную маску вокруг кликнутого пикселя.
            Основная маска строится по простой идее:
            - берём яркость пикселя под кликом;
            - оцениваем локальный допуск по шуму вокруг клика;
            - выбираем похожие пиксели с той же температурной полярностью.
        """
        return self.build_mask_with_tolerance_scale(patch=patch, click_x=click_x, click_y=click_y, tolerance_scale=1.0)

    def build_mask_with_tolerance_scale(
        self,
        patch: np.ndarray,
        click_x: int,
        click_y: int,
        *,
        tolerance_scale: float,
    ) -> tuple[np.ndarray, TargetPolarity]:
        """
            Построить маску с управляемым допуском по яркости.
            `tolerance_scale` нужен для повторных попыток: если первая маска
            захватила слишком большую область, можно ужать допуск и выделить более компактное ядро объекта.
        """
        clicked_value = int(patch[click_y, click_x])
        median_value = float(np.median(patch))

        tolerance = self.estimate_tolerance(patch=patch, click_x=click_x, click_y=click_y)
        tolerance = max(self.config.min_tolerance, int(round(tolerance * tolerance_scale)))

        # similarity_mask оставляет пиксели, близкие к кликнутому пикселю по яркости.
        patch_int = patch.astype(np.int16)
        similarity_mask = np.abs(patch_int - clicked_value) <= tolerance

        # По медиане патча определяем, цель горячее или холоднее окружения.
        # Это важно, чтобы не захватить контрастные пиксели противоположной природы.
        hot_object = clicked_value >= median_value

        if hot_object:
            intensity_mask = patch >= clicked_value - tolerance
            polarity = TargetPolarity.HOT
        else:
            intensity_mask = patch <= clicked_value + tolerance
            polarity = TargetPolarity.COLD

        # Финальная маска требует одновременно похожести по яркости и правильной полярности.
        mask = np.logical_and(similarity_mask, intensity_mask).astype(np.uint8) * 255

        # Если морфология/порог случайно выбили сам клик, откатываемся к простой похожести.
        # Клик должен оставаться seed-точкой, иначе connected components дальше теряет объект.
        if mask[click_y, click_x] == 0:
            mask = similarity_mask.astype(np.uint8) * 255

        return self._clean_mask(mask), polarity

    def build_score_mask(self, patch: np.ndarray, click_x: int, click_y: int) -> np.ndarray:
        """
            Построить резервную маску, если обычная похожесть не сработала.
            Резервная маска использует не только разницу яркости с кликом,
            но и локальные границы, а также расстояние до клика.
            Это помогает не провалиться, когда объект слабоконтрастный или шумный.
        """
        clicked_value = float(patch[click_y, click_x])
        tolerance = float(self.estimate_tolerance(patch=patch, click_x=click_x, click_y=click_y))

        # Размытие снижает влияние одиночных шумовых пикселей перед Sobel.
        blurred = cv2.GaussianBlur(patch, (self.config.score_blur_kernel, self.config.score_blur_kernel),0)

        # Градиент помогает учитывать локальные границы объекта.
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=self.config.score_sobel_kernel)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=self.config.score_sobel_kernel)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Чем дальше пиксель от клика, тем менее охотно он должен попадать в объект.
        yy, xx = np.indices(patch.shape)
        distance = np.sqrt((xx - click_x) ** 2 + (yy - click_y) ** 2)
        distance /= max(float(max(patch.shape)), 1.0)

        # difference показывает, насколько пиксель похож на кликнутый по яркости.
        difference = np.abs(patch.astype(np.float32) - clicked_value) / max(tolerance, 1.0)

        # Чем ниже score, тем пиксель ближе к ожидаемой области объекта.
        score = (
            difference * self.config.score_difference_weight
            + gradient * self.config.score_gradient_weight
            + distance * self.config.score_distance_weight
        )
        mask = (score <= self.config.score_threshold).astype(np.uint8) * 255

        return self._clean_mask(mask)

    def estimate_tolerance(self, patch: np.ndarray, click_x: int, click_y: int) -> int:
        """
            Оценить допустимое отклонение яркости от кликнутого пикселя.
            Допуск строится от локального стандартного отклонения: шумный участок
            получает больший tolerance, спокойный участок — более строгий.
        """
        local = self.local_window(patch=patch, click_x=click_x, click_y=click_y)
        local_std = float(np.std(local) + 1.0)

        tolerance = int(round(local_std * self.config.similarity_sigma + 4.0))
        tolerance = max(self.config.min_tolerance, tolerance)
        tolerance = min(self.config.max_tolerance, tolerance)

        return tolerance

    def local_window(self, patch: np.ndarray, click_x: int, click_y: int) -> np.ndarray:
        """Вернуть маленькое окно вокруг клика для локальной статистики."""
        radius = self.config.local_window_radius
        return patch[max(0, click_y - radius): click_y + radius + 1, max(0, click_x - radius): click_x + radius + 1]

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
            Очистить бинарную маску морфологическими операциями.
            Opening убирает мелкие шумовые точки.
            Closing закрывает небольшие разрывы внутри найденной области.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.mask_morphology_kernel, self.config.mask_morphology_kernel),
        )

        if self.config.mask_open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.mask_open_iterations)

        if self.config.mask_close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.mask_close_iterations)

        return mask