import logging

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from extractors import FeatureExtractor

logger = logging.getLogger(__name__)


def hellinger():
    """Квадратный корень из L1-гистограммы: евклидово расстояние между такими векторами
    соответствует расстоянию Хеллингера между гистограммами, редкие слова получают больший вес."""
    return FunctionTransformer(np.sqrt)


CLASSIFIERS = {
    'linear_svm': lambda seed: make_pipeline(hellinger(), StandardScaler(), LinearSVC(C=0.01, max_iter=5000,
                                                                                      random_state=seed)),
    'svm': lambda seed: make_pipeline(hellinger(), StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale',
                                                                         random_state=seed)),
    'logreg': lambda seed: make_pipeline(hellinger(), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
    'rf': lambda seed: RandomForestClassifier(n_estimators=500, random_state=seed),
    'knn': lambda seed: make_pipeline(hellinger(), KNeighborsClassifier(n_neighbors=7, weights='distance')),
}


class BagOfWords:
    """Мешок визуальных слов: словарь (центры K-средних), PCA-проекция дескрипторов, классификатор."""

    def __init__(self, extractor, vocab_size=1000, classifier='linear_svm', pca_dim=None, seed=42):
        if vocab_size < 2:
            raise ValueError('Размер словаря должен быть не меньше 2')
        if classifier not in CLASSIFIERS:
            raise ValueError(f'Неизвестный классификатор: {classifier}. Доступны: {", ".join(CLASSIFIERS)}')
        self.extractor = extractor
        self.vocab_size = vocab_size
        self.classifier_name = classifier
        self.pca_dim = pca_dim
        self.seed = seed
        self.vocabulary = None #K x D
        self.pca_mean = None
        self.pca_vectors = None
        self.classifier = None
        self.class_names = None

    def project(self, descriptors):
        """Проекция дескрипторов на главные компоненты: y = E (x - mean)."""
        if self.pca_vectors is None or len(descriptors) == 0:
            return descriptors
        return cv2.PCAProject(descriptors, self.pca_mean, self.pca_vectors)

    def build_vocabulary(self, descriptors):
        """Строит словарь из vocab_size визуальных слов методом K-средних по матрице дескрипторов M x D."""
        descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        if len(descriptors) < self.vocab_size:
            raise ValueError(f'Дескрипторов ({len(descriptors)}) меньше, чем слов в словаре ({self.vocab_size})')
        if self.pca_dim and self.pca_dim < descriptors.shape[1]:
            self.pca_mean, self.pca_vectors, _ = cv2.PCACompute2(descriptors, mean=None, maxComponents=self.pca_dim)
            descriptors = self.project(descriptors)
            logger.info('PCA: размерность %d -> %d', self.pca_vectors.shape[1], self.pca_dim)

        logger.info('K-средних: %d дескрипторов размерности %d, K = %d', *descriptors.shape, self.vocab_size)
        cv2.setRNGSeed(self.seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        _, _, centers = cv2.kmeans(descriptors, self.vocab_size, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        self.vocabulary = centers.astype(np.float32)

    def assign_words(self, descriptors):
        """Номер ближайшего визуального слова для каждого дескриптора.
        |x - c|^2 = |x|^2 - 2 x c + |c|^2, слагаемое |x|^2 на выбор слова не влияет."""
        if self.vocabulary is None:
            raise RuntimeError('Словарь не построен')
        if len(descriptors) == 0:
            return np.empty(0, dtype=np.int64)
        x = self.project(np.asarray(descriptors, dtype=np.float32))
        distances = (self.vocabulary ** 2).sum(axis=1) - 2 * x @ self.vocabulary.T
        return distances.argmin(axis=1)

    def compute_histogram(self, descriptors):
        """BoW-гистограмма: доля дескрипторов изображения, попавших в каждое слово (сумма равна 1).
        Изображение без дескрипторов дает нулевой вектор."""
        counts = np.bincount(self.assign_words(descriptors), minlength=self.vocab_size).astype(np.float32)
        total = counts.sum()
        return counts / total if total > 0 else counts

    def train(self, histograms, labels, class_names):
        """Обучает классификатор на матрице гистограмм обучающей выборки."""
        self.class_names = list(class_names)
        self.classifier = CLASSIFIERS[self.classifier_name](self.seed)
        logger.info('Обучение классификатора %s на %d гистограммах', self.classifier_name, len(histograms))
        self.classifier.fit(histograms, labels)

    def predict_histograms(self, histograms):
        """Индексы предсказанных классов для матрицы гистограмм."""
        if self.classifier is None:
            raise RuntimeError('Классификатор не обучен')
        return self.classifier.predict(histograms)

    def predict(self, image):
        """Классифицирует изображение. Возвращает имя класса, гистограмму, ключевые точки и номера их слов."""
        keypoints, descriptors = self.extractor.detect_and_compute(image)
        words = self.assign_words(descriptors)
        histogram = self.compute_histogram(descriptors)
        label = int(self.predict_histograms(histogram[None, :])[0])
        return self.class_names[label], histogram, keypoints, words

    def save_model(self, path):
        """Сохраняет в один файл параметры экстрактора, словарь, PCA и обученный классификатор."""
        joblib.dump({
            'extractor': self.extractor.name,
            'extractor_params': self.extractor.get_params(),
            'vocab_size': self.vocab_size,
            'classifier_name': self.classifier_name,
            'pca_dim': self.pca_dim,
            'seed': self.seed,
            'vocabulary': self.vocabulary,
            'pca_mean': self.pca_mean,
            'pca_vectors': self.pca_vectors,
            'classifier': self.classifier,
            'class_names': self.class_names,
        }, path)
        logger.info('Модель сохранена: %s', path)

    @staticmethod
    def load_model(path):
        """Загружает модель и создает экстрактор с сохраненными параметрами через FeatureExtractor.create."""
        state = joblib.load(path)
        extractor = FeatureExtractor.create(state['extractor'], **state['extractor_params'])
        model = BagOfWords(extractor, state['vocab_size'], state['classifier_name'], state['pca_dim'], state['seed'])
        for key in ('vocabulary', 'pca_mean', 'pca_vectors', 'classifier', 'class_names'):
            setattr(model, key, state[key])
        logger.info('Модель загружена: %s (метод %s, K = %d)', path, extractor.name, model.vocab_size)
        return model
