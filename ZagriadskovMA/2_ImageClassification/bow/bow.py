"""Мешок визуальных слов: словарь (K-средних), BoW-гистограммы и классификатор."""

import logging
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .extractors import FeatureExtractor

logger = logging.getLogger(__name__)

# Модуль cv2.ml в OpenCV 5 перенесен в opencv-contrib, поэтому классификаторы берутся из scikit-learn
CLASSIFIERS = {
    "svm": lambda seed: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale", random_state=seed)),
    "linear_svm": lambda seed: make_pipeline(StandardScaler(), LinearSVC(C=0.1, random_state=seed)),
    "logreg": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "rf": lambda seed: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed),
    "knn": lambda seed: KNeighborsClassifier(n_neighbors=5),
}


class BagOfWords:
    """Модель "мешок визуальных слов".

    Хранит экстрактор признаков, словарь визуальных слов (центроиды K-средних),
    и обученный классификатор.
    """

    def __init__(self, extractor, vocab_size=200, classifier="svm",
                 max_descriptors=200_000, random_state=42):
        if classifier not in CLASSIFIERS:
            raise ValueError(f"Неизвестный классификатор '{classifier}'. Доступны: {sorted(CLASSIFIERS)}")
        self.extractor = extractor
        self.vocab_size = vocab_size
        self.classifier_name = classifier
        self.max_descriptors = max_descriptors
        self.random_state = random_state

        self.vocabulary = None   # K x D', float32
        self.classifier = None
        self.class_names = None
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    # ---------- словарь ----------

    def build_vocabulary(self, descriptor_list):
        """Строит словарь визуальных слов методом MiniBatchKMeans.

        :param descriptor_list: список матриц дескрипторов (по одной на изображение).
        """
        nonempty = [d for d in descriptor_list if len(d) > 0]
        if not nonempty:
            raise ValueError("Не найдено дескрипторов для словаря")
        all_descriptors = np.vstack(nonempty).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        if len(all_descriptors) > self.max_descriptors:
            idx = rng.choice(len(all_descriptors), self.max_descriptors, replace=False)
            all_descriptors = all_descriptors[idx]
        logger.info("K-средних: %d дескрипторов размерности %d, K=%d",
                    len(all_descriptors), all_descriptors.shape[1], self.vocab_size)

        if len(all_descriptors) < self.vocab_size:
            raise ValueError("Число дескрипторов меньше размера словаря")
        start = time.perf_counter()
        clustering = MiniBatchKMeans(n_clusters=self.vocab_size, batch_size=1024,
                                     n_init=3, max_iter=100, max_no_improvement=10,
                                     random_state=self.random_state)
        clustering.fit(all_descriptors)
        self.vocabulary = clustering.cluster_centers_.astype(np.float32)
        logger.info("MiniBatchKMeans: %.1f с", time.perf_counter() - start)
        return self.vocabulary

    # ---------- гистограммы ----------

    def compute_histogram(self, descriptors):
        """L1-нормированная гистограмма визуальных слов всего изображения."""
        if self.vocabulary is None:
            raise RuntimeError("Словарь не построен")
        hist = np.zeros(self.vocab_size, dtype=np.float32)
        if len(descriptors) == 0:
            return hist
        matches = self._matcher.match(descriptors.astype(np.float32), self.vocabulary)
        words = np.fromiter((m.trainIdx for m in matches), dtype=np.int64)
        hist = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
        return hist / hist.sum()

    def compute_histograms(self, descriptor_list):
        return np.vstack([self.compute_histogram(d) for d in descriptor_list])

    # ---------- обучение и предсказание ----------

    def fit_classifier(self, histograms, labels, class_names, fit_indices=None, validation_indices=None):
        """Обучает классификатор на BoW-гистограммах обучающей выборки.

        """
        self.class_names = list(class_names)
        logger.info("Обучение классификатора '%s' на %d гистограммах", self.classifier_name, len(histograms))
        self.classifier = CLASSIFIERS[self.classifier_name](self.random_state)
        labels = np.asarray(labels)
        if validation_indices is not None and self.classifier_name == "svm":
            best_score, best = -1.0, None
            # Словарь построен только на fit_indices; test здесь не используется.
            for c in (0.1, 1.0, 10.0):
                for gamma in (0.1 / self.vocab_size, 1.0 / self.vocab_size, "scale"):
                    candidate = clone(self.classifier).set_params(svc__C=c, svc__gamma=gamma)
                    candidate.fit(histograms[fit_indices], labels[fit_indices])
                    score = balanced_accuracy_score(labels[validation_indices],
                                                     candidate.predict(histograms[validation_indices]))
                    logger.info("Validation balanced accuracy: %.4f (C=%s, gamma=%s)", score, c, gamma)
                    if score > best_score:
                        best_score, best = score, candidate
            logger.info("Выбраны параметры SVM: %s; validation=%.4f",
                        best.named_steps["svc"].get_params(), best_score)
            self.classifier = clone(best)
        self.classifier.fit(histograms, labels)
        return histograms

    def predict_histograms(self, histograms):
        """Возвращает индексы классов для матрицы гистограмм."""
        return self.classifier.predict(histograms)

    def predict(self, image):
        """Классифицирует одно изображение.

        :return: (имя класса, BoW-гистограмма, ключевые точки).
        """
        keypoints, descriptors = self.extractor.detect_and_compute(image)
        hist = self.compute_histogram(descriptors)
        label = int(self.predict_histograms(hist[None, :])[0])
        return self.class_names[label], hist, keypoints

    # ---------- сохранение ----------

    def save_model(self, path):
        """Сохраняет словарь, классификатор и параметры экстрактора в один файл (joblib)."""
        state = {
            "format_version": 2,
            "extractor_name": self.extractor.name,
            "extractor_params": self.extractor.get_params(),
            "vocab_size": self.vocab_size,
            "classifier_name": self.classifier_name,
            "max_descriptors": self.max_descriptors,
            "random_state": self.random_state,
            "vocabulary": self.vocabulary,
            "classifier": self.classifier,
            "class_names": self.class_names,
        }
        joblib.dump(state, path)
        logger.info("Модель сохранена: %s", path)

    @classmethod
    def load_model(cls, path, **extractor_overrides):
        """Загружает модель и заново создает экстрактор через фабрику FeatureExtractor.create.

        extractor_overrides позволяют заменить параметры экстрактора (например, device);
        параметры, которых у экстрактора нет, игнорируются.
        """
        state = joblib.load(path)
        if state.get("format_version") != 2:
            raise ValueError("Устаревший формат модели: переобучите модель")
        if state.get("spatial_pyramid", False):
            raise ValueError("Модель с пространственной пирамидой больше не поддерживается; переобучите модель")
        params = dict(state["extractor_params"])
        params.update({k: v for k, v in extractor_overrides.items() if k in params})
        extractor = FeatureExtractor.create(state["extractor_name"], **params)
        model = cls(extractor, vocab_size=state["vocab_size"], classifier=state["classifier_name"],
                    max_descriptors=state["max_descriptors"],
                    random_state=state["random_state"])
        for key in ("vocabulary", "classifier", "class_names"):
            setattr(model, key, state[key])
        logger.info("Модель загружена: %s (экстрактор %s, K=%d)", path, extractor.name, model.vocab_size)
        return model
