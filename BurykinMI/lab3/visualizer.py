import os
from typing import Optional, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from abstract import FeatureExtractor


# ============================================================================
# Модуль для визуализации работы детекторов признаков.
# Класс KeypointVisualizer предоставляет методы:
# - visualize_keypoints: отрисовка ключевых точек на одном изображении
# - compare_detectors: сравнение разных детекторов (SIFT, ORB, AKAZE) на одной картинке
# - visualize_matches: показ совпадающих точек между двумя изображениями
# Использует OpenCV для отрисовки и matplotlib для отображения результатов.
# ============================================================================

class KeypointVisualizer:
    """Класс для визуализации ключевых точек и дескрипторов"""

    def __init__(self, feature_extractor: FeatureExtractor):
        """
        Args:
            feature_extractor: Экстрактор признаков
        """
        self.feature_extractor = feature_extractor

    def visualize_keypoints(
            self,
            image_path: str,
            save_path: Optional[str] = None,
            show: bool = True,
            max_keypoints: Optional[int] = None
    ) -> np.ndarray:
        """
        Визуализировать ключевые точки на изображении

        Args:
            image_path: Путь к изображению
            save_path: Путь для сохранения результата (опционально)
            show: Показать изображение с помощью matplotlib
            max_keypoints: Максимальное количество точек для отображения (None = все)

        Returns:
            Изображение с нарисованными ключевыми точками
        """
        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Получение детектора и извлечение ключевых точек
        detector = self.feature_extractor.detector
        keypoints, descriptors = detector.detectAndCompute(gray, None)

        # Ограничение количества точек, если задано
        if max_keypoints and len(keypoints) > max_keypoints:
            # Сортировка по response (сила ключевой точки)
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
            keypoints = keypoints[:max_keypoints]

        # Рисование ключевых точек
        img_with_keypoints = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            color=(0, 255, 0)
        )

        # Добавление информации о детекторе и количестве точек
        font = cv2.FONT_HERSHEY_SIMPLEX
        detector_name = self.feature_extractor.get_name().upper()
        text = f"{detector_name}: {len(keypoints)} keypoints"

        cv2.putText(
            img_with_keypoints,
            text,
            (10, 30),
            font,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Сохранение результата
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_with_keypoints)
            print(f"Визуализация сохранена в {save_path}")

        # Отображение результата
        if show:
            self._show_image(img_with_keypoints, f"Ключевые точки ({detector_name})")

        return img_with_keypoints

    def compare_detectors(
            self,
            image_path: str,
            extractors: List[FeatureExtractor],
            save_path: Optional[str] = None,
            show: bool = True,
            max_keypoints: Optional[int] = None
    ) -> None:
        """
        Сравнить разные детекторы на одном изображении

        Args:
            image_path: Путь к изображению
            extractors: Список экстракторов для сравнения
            save_path: Путь для сохранения результата (опционально)
            show: Показать изображение с помощью matplotlib
            max_keypoints: Максимальное количество точек для каждого детектора
        """
        n_extractors = len(extractors)
        fig, axes = plt.subplots(1, n_extractors, figsize=(6 * n_extractors, 6))

        if n_extractors == 1:
            axes = [axes]

        # Загрузка исходного изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        for idx, extractor in enumerate(extractors):
            # Создание временного визуализатора для каждого экстрактора
            temp_visualizer = KeypointVisualizer(extractor)
            img_with_kp = temp_visualizer.visualize_keypoints(
                image_path,
                save_path=None,
                show=False,
                max_keypoints=max_keypoints
            )

            # Конвертация BGR -> RGB для matplotlib
            img_rgb = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)

            axes[idx].imshow(img_rgb)
            axes[idx].set_title(extractor.get_name().upper())
            axes[idx].axis('off')

        plt.tight_layout()

        # Сохранение результата
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Сравнение сохранено в {save_path}")

        # Отображение результата
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_matches(
            self,
            image_path1: str,
            image_path2: str,
            save_path: Optional[str] = None,
            show: bool = True,
            max_matches: int = 50
    ) -> None:
        """
        Визуализировать совпадающие ключевые точки между двумя изображениями

        Args:
            image_path1: Путь к первому изображению
            image_path2: Путь ко второму изображению
            save_path: Путь для сохранения результата (опционально)
            show: Показать изображение с помощью matplotlib
            max_matches: Максимальное количество совпадений для отображения
        """
        # Загрузка изображений
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)

        if img1 is None or img2 is None:
            raise ValueError("Не удалось загрузить одно из изображений")

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Извлечение ключевых точек и дескрипторов
        detector = self.feature_extractor.detector
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            print("Не удалось извлечь дескрипторы из одного из изображений")
            return

        # Создание матчера
        if self.feature_extractor.get_name() == 'sift':
            # Для SIFT используем FLANN (быстрее для float дескрипторов)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Для бинарных дескрипторов (ORB, AKAZE) используем BFMatcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Поиск совпадений
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

        # Рисование совпадений
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Добавление информации
        font = cv2.FONT_HERSHEY_SIMPLEX
        detector_name = self.feature_extractor.get_name().upper()
        text = f"{detector_name}: {len(matches)} matches"
        cv2.putText(img_matches, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Сохранение результата
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_matches)
            print(f"Визуализация совпадений сохранена в {save_path}")

        # Отображение результата
        if show:
            self._show_image(img_matches, f"Совпадения ключевых точек ({detector_name})")

    @staticmethod
    def _show_image(img: np.ndarray, title: str) -> None:
        """Показать изображение с помощью matplotlib"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
