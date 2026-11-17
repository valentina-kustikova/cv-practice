from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .extractors import FeatureExtractor

logger = logging.getLogger(__name__)

SUPPORTED_CLASSIFIERS = ("svm", "rf", "knn")
SUPPORTED_NORMS = ("l1", "l2", "none")

MODEL_FORMAT = "bow-model-v1"


class BagOfWords:
    """Модель BoW: словарь визуальных слов (K-средних) + классификатор поверх гистограмм.

    :param extractor: извлекатель признаков (наследник FeatureExtractor)
    :param vocab_size: размер словаря визуальных слов (число кластеров K)
    :param classifier_type: метод машинного обучения: 'svm' | 'rf' | 'knn'
    :param hist_norm: нормировка гистограмм: 'l1' | 'l2' | 'none'
    :param random_state: зерно случайности (воспроизводимость)
    """

    def __init__(self, extractor: FeatureExtractor, vocab_size: int = 200,
                 classifier_type: str = "svm", hist_norm: str = "l1",
                 random_state: int = 42) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size должен быть не меньше 2")
        if classifier_type not in SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Неизвестный классификатор: '{classifier_type}'. Доступно: {SUPPORTED_CLASSIFIERS}")
        if hist_norm not in SUPPORTED_NORMS:
            raise ValueError(f"Неизвестная нормировка: '{hist_norm}'. Доступно: {SUPPORTED_NORMS}")
        self.extractor = extractor
        self.vocab_size = int(vocab_size)
        self.classifier_type = classifier_type
        self.hist_norm = hist_norm
        self.random_state = int(random_state)
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.classifier: Optional[Any] = None
        self.class_names: List[str] = []


    def build_vocabulary(self, descriptors_list: Sequence[np.ndarray], sample_size: int = 0) -> None:
        """Построить словарь визуальных слов: кластеризация дескрипторов методом K-средних.

        :param descriptors_list: список матриц дескрипторов (N_i, D) по всем обучающим изображениям
        :param sample_size: если > 0 и дескрипторов больше — строить словарь по случайной
            подвыборке указанного размера (ускоряет K-средних почти без потери качества)
        """
        non_empty = [np.asarray(d, dtype=np.float32) for d in descriptors_list if len(d) > 0]
        if not non_empty:
            raise ValueError("Не извлечено ни одного дескриптора — проверьте изображения и экстрактор")
        all_descriptors = np.vstack(non_empty)
        total = all_descriptors.shape[0]
        logger.info("Всего дескрипторов для словаря: %d (размерность %d)", total, all_descriptors.shape[1])
        if sample_size and total > sample_size:
            rng = np.random.default_rng(self.random_state)
            all_descriptors = all_descriptors[rng.choice(total, size=sample_size, replace=False)]
            logger.info("Для построения словаря взята случайная подвыборка: %d дескрипторов", sample_size)
        if all_descriptors.shape[0] < self.vocab_size:
            logger.warning(
                "Дескрипторов (%d) меньше, чем запрошенных слов словаря (%d), — размер словаря уменьшен",
                all_descriptors.shape[0], self.vocab_size,
            )
            self.vocab_size = max(2, all_descriptors.shape[0])
        batch_size = int(min(1024, all_descriptors.shape[0]))
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=batch_size,
            init_size=min(3 * batch_size, all_descriptors.shape[0]),
            n_init=3,
            max_iter=200,
            random_state=self.random_state,
        )
        logger.info("Кластеризация K-средних: %d кластеров по %d дескрипторам...", self.vocab_size,
                    all_descriptors.shape[0])
        self.kmeans.fit(all_descriptors)
        logger.info("Словарь построен: %d визуальных слов (инерция %.1f)", self.vocab_size,
                    float(self.kmeans.inertia_))


    def compute_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Вычислить нормированную BoW-гистограмму для матрицы дескрипторов (N, D).

        Каждый дескриптор относился к ближайшему визуальному слову (кластеру),
        затем строится гистограмма частот слов и нормируется.

        :param descriptors: матрица дескрипторов; если дескрипторов нет — возвращается нулевой вектор
        :return: вектор длины vocab_size
        """
        self._check_vocabulary()
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.size == 0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        words = self.kmeans.predict(descriptors)
        histogram = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
        return self._normalize(histogram)

    def histogram_from_image(self, image: np.ndarray) -> np.ndarray:
        """Извлечь дескрипторы изображения и вычислить его BoW-гистограмму."""
        return self.compute_histogram(self.extractor.detect_and_compute(image))

    def _normalize(self, histogram: np.ndarray) -> np.ndarray:
        """Нормировка гистограммы: 'l1' (сумма = 1), 'l2' (длина = 1) или без нормировки."""
        if self.hist_norm == "l1":
            total = histogram.sum()
            return histogram / total if total > 0 else histogram
        if self.hist_norm == "l2":
            norm = float(np.linalg.norm(histogram))
            return histogram / norm if norm > 0 else histogram
        return histogram


    def train_classifier(self, histograms: Sequence[np.ndarray], labels: Sequence[int]) -> float:
        """Обучить классификатор на BoW-гистограммах; вернуть точность на обучающей выборке.

        :param histograms: список гистограмм (векторы длины vocab_size)
        :param labels: метки классов (индексы, согласованы с class_names)
        """
        self._check_vocabulary()
        x = np.vstack([np.asarray(h, dtype=np.float32) for h in histograms])
        y = np.asarray(labels)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("Некорректные данные для обучения классификатора")
        self.classifier = self._make_classifier()
        logger.info("Обучение классификатора '%s': %d объектов, %d признаков...", self.classifier_type, *x.shape)
        self.classifier.fit(x, y)
        train_accuracy = float(self.classifier.score(x, y))
        logger.info("Классификатор обучен. Точность на обучающей выборке: %.2f%%", train_accuracy * 100)
        return train_accuracy

    def _make_classifier(self):
        """Создать классификатор выбранного типа с разумными настройками по умолчанию."""
        if self.classifier_type == "svm":
            return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=self.random_state)
        if self.classifier_type == "rf":
            return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=self.random_state)
        return KNeighborsClassifier(n_neighbors=5, weights="distance")

    def predict_histogram(self, histogram: np.ndarray) -> int:
        """Предсказать класс изображения по его BoW-гистограмме (возвращает индекс класса)."""
        self._check_classifier()
        x = np.asarray(histogram, dtype=np.float32).reshape(1, -1)
        return int(self.classifier.predict(x)[0])

    def predict_image(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Классифицировать изображение; возвращает (индекс класса, BoW-гистограмма)."""
        histogram = self.histogram_from_image(image)
        return self.predict_histogram(histogram), histogram

    def predict(self, image: np.ndarray) -> int:
        """Предсказать класс изображения (индекс класса)."""
        return self.predict_histogram(self.histogram_from_image(image))

    def predict_proba_histogram(self, histogram: np.ndarray) -> Optional[np.ndarray]:
        """Оценки вероятностей классов по гистограмме (None, если классификатор их не поддерживает)."""
        self._check_classifier()
        x = np.asarray(histogram, dtype=np.float32).reshape(1, -1)
        if hasattr(self.classifier, "predict_proba"):
            return np.asarray(self.classifier.predict_proba(x)[0], dtype=np.float32)
        return None


    def save_model(self, path) -> None:

        self._check_vocabulary()
        self._check_classifier()
        payload: Dict[str, Any] = {
            "format": MODEL_FORMAT,
            "vocab_size": self.vocab_size,
            "classifier_type": self.classifier_type,
            "hist_norm": self.hist_norm,
            "random_state": self.random_state,
            "kmeans": self.kmeans,
            "classifier": self.classifier,
            "extractor": {"name": self.extractor.name, "params": self.extractor.get_params()},
            "class_names": list(self.class_names),
        }
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Модель сохранена: %s", model_path)

    @classmethod
    def load_model(cls, path) -> "BagOfWords":
        """Загрузить модель из файла, созданного save_model (экстрактор воссоздаётся автоматически).

        :raises FileNotFoundError: если файла модели нет
        :raises ValueError: если файл не является моделью этого проекта
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_path.resolve()}. Сначала обучите модель (режим train)."
            )
        with open(model_path, "rb") as file:
            payload = pickle.load(file)
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"Файл {model_path} не является моделью BoW этого проекта")
        extractor = FeatureExtractor.create(payload["extractor"]["name"], **payload["extractor"].get("params", {}))
        bow = cls(extractor, payload["vocab_size"], payload["classifier_type"],
                  payload["hist_norm"], payload.get("random_state", 42))
        bow.kmeans = payload["kmeans"]
        bow.classifier = payload["classifier"]
        bow.class_names = list(payload.get("class_names", []))
        logger.info("Модель загружена: %s (экстрактор '%s', словарь %d слов, классификатор '%s', классы: %d)",
                    model_path, bow.extractor.name, bow.vocab_size, bow.classifier_type, len(bow.class_names))
        return bow

    def _check_vocabulary(self) -> None:
        if self.kmeans is None:
            raise RuntimeError("Словарь визуальных слов не построен — выполните build_vocabulary или load_model")

    def _check_classifier(self) -> None:
        if self.classifier is None:
            raise RuntimeError("Классификатор не обучен — выполните train_classifier или load_model")
