"""Мешок визуальных слов: словарь (K-средних), BoW-гистограммы и классификатор."""

import logging

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
    "svm": lambda seed: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)),
    "linear_svm": lambda seed: make_pipeline(StandardScaler(), LinearSVC(C=0.1, random_state=seed)),
    "logreg": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "rf": lambda seed: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed),
    "knn": lambda seed: KNeighborsClassifier(n_neighbors=5),
}


class BagOfWords:
    """Модель "мешок визуальных слов".

    Хранит экстрактор признаков, словарь визуальных слов (центроиды K-средних),
    опциональную PCA-проекцию дескрипторов, IDF-веса и обученный классификатор.
    """

    def __init__(self, extractor, vocab_size=200, classifier="svm", pca_dim=None,
                 max_descriptors=200_000, tfidf=False, random_state=42):
        if classifier not in CLASSIFIERS:
            raise ValueError(f"Неизвестный классификатор '{classifier}'. Доступны: {sorted(CLASSIFIERS)}")
        self.extractor = extractor
        self.vocab_size = vocab_size
        self.classifier_name = classifier
        self.pca_dim = pca_dim
        self.max_descriptors = max_descriptors
        self.tfidf = tfidf
        self.random_state = random_state

        self.vocabulary = None   # K x D', float32
        self.pca_mean = None
        self.pca_eigenvectors = None
        self.idf = None
        self.classifier = None
        self.class_names = None
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    # ---------- словарь ----------

    def _project(self, descriptors):
        if self.pca_eigenvectors is None or len(descriptors) == 0:
            return descriptors
        return cv2.PCAProject(descriptors, self.pca_mean, self.pca_eigenvectors)

    def build_vocabulary(self, descriptor_list):
        """Строит словарь визуальных слов методом K-средних (cv2.kmeans).

        :param descriptor_list: список матриц дескрипторов (по одной на изображение).
        """
        all_descriptors = np.vstack([d for d in descriptor_list if len(d) > 0]).astype(np.float32)
        rng = np.random.default_rng(self.random_state)
        if len(all_descriptors) > self.max_descriptors:
            idx = rng.choice(len(all_descriptors), self.max_descriptors, replace=False)
            all_descriptors = all_descriptors[idx]
        logger.info("K-средних: %d дескрипторов размерности %d, K=%d",
                    len(all_descriptors), all_descriptors.shape[1], self.vocab_size)

        if self.pca_dim and self.pca_dim < all_descriptors.shape[1]:
            self.pca_mean, self.pca_eigenvectors, _ = cv2.PCACompute2(
                all_descriptors, mean=None, maxComponents=self.pca_dim)
            all_descriptors = self._project(all_descriptors)
            logger.info("PCA: размерность понижена до %d", self.pca_dim)

        cv2.setRNGSeed(self.random_state)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        _, _, centers = cv2.kmeans(all_descriptors, self.vocab_size, None, criteria,
                                   attempts=3, flags=cv2.KMEANS_PP_CENTERS)
        self.vocabulary = centers.astype(np.float32)
        return self.vocabulary

    # ---------- гистограммы ----------

    def compute_histogram(self, descriptors):
        """Вычисляет нормализованную (L1) BoW-гистограмму по дескрипторам изображения.

        Каждый дескриптор относится к ближайшему визуальному слову (cv2.BFMatcher),
        затем считается частота слов. Изображение без дескрипторов дает нулевой вектор.
        """
        if self.vocabulary is None:
            raise RuntimeError("Словарь не построен: вызовите build_vocabulary или load_model")
        hist = np.zeros(self.vocab_size, dtype=np.float32)
        if len(descriptors) > 0:
            matches = self._matcher.match(self._project(descriptors.astype(np.float32)), self.vocabulary)
            words = np.fromiter((m.trainIdx for m in matches), dtype=np.int64, count=len(matches))
            hist = np.bincount(words, minlength=self.vocab_size).astype(np.float32)
        if self.idf is not None:
            hist *= self.idf
        total = hist.sum()
        return hist / total if total > 0 else hist

    def compute_histograms(self, descriptor_list):
        return np.vstack([self.compute_histogram(d) for d in descriptor_list])

    # ---------- обучение и предсказание ----------

    def fit_classifier(self, histograms, labels, class_names):
        """Обучает классификатор на BoW-гистограммах обучающей выборки.

        Гистограммы должны быть вычислены compute_histogram до задания IDF-весов;
        при tfidf=True веса считаются здесь и применяются к гистограммам.
        """
        self.class_names = list(class_names)
        if self.tfidf:
            # Сглаженный IDF (как в sklearn TfidfTransformer): idf = ln((1 + N) / (1 + df)) + 1.
            # Слова, встречающиеся почти во всех изображениях, получают вес около 1,
            # редкие – больше; вес всегда положителен, деления на ноль нет
            n = len(histograms)
            df = np.count_nonzero(histograms > 0, axis=0)
            self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
            histograms = histograms * self.idf
            totals = histograms.sum(axis=1, keepdims=True)
            histograms = np.divide(histograms, totals, out=np.zeros_like(histograms), where=totals > 0)

        logger.info("Обучение классификатора '%s' на %d гистограммах", self.classifier_name, len(histograms))
        self.classifier = CLASSIFIERS[self.classifier_name](self.random_state)
        self.classifier.fit(histograms, np.asarray(labels))
        return histograms  # с учетом IDF-весов, если они заданы

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
            "extractor_name": self.extractor.name,
            "extractor_params": self.extractor.get_params(),
            "vocab_size": self.vocab_size,
            "classifier_name": self.classifier_name,
            "pca_dim": self.pca_dim,
            "max_descriptors": self.max_descriptors,
            "tfidf": self.tfidf,
            "random_state": self.random_state,
            "vocabulary": self.vocabulary,
            "pca_mean": self.pca_mean,
            "pca_eigenvectors": self.pca_eigenvectors,
            "idf": self.idf,
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
        params = dict(state["extractor_params"])
        params.update({k: v for k, v in extractor_overrides.items() if k in params})
        extractor = FeatureExtractor.create(state["extractor_name"], **params)
        model = cls(extractor, vocab_size=state["vocab_size"], classifier=state["classifier_name"],
                    pca_dim=state["pca_dim"], max_descriptors=state["max_descriptors"],
                    tfidf=state["tfidf"], random_state=state["random_state"])
        for key in ("vocabulary", "pca_mean", "pca_eigenvectors", "idf", "classifier", "class_names"):
            setattr(model, key, state[key])
        logger.info("Модель загружена: %s (экстрактор %s, K=%d)", path, extractor.name, model.vocab_size)
        return model
