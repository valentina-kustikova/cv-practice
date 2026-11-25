"""Строит словарь визуальных слов и определяет класс изображения."""
import logging
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dataset import read_image



class BagOfWords:
    """Строит словарь визуальных слов и классифицирует снимки по их гистограммам."""

    FORMAT_VERSION = 1

    def __init__(self, n_words=128, classifier='svm', seed=42, class_weight=None, C=1.0):
        """Настраивает K-Means и выбранный классификатор."""
        if n_words < 1 or classifier not in {'svm', 'rf', 'lr'} or class_weight not in {None, 'balanced'}:
            raise ValueError('Неверные параметры словаря или классификатора')
        if C <= 0:
            raise ValueError('C должно быть положительным')
        self.C = C
        self.n_words, self.seed = n_words, seed
        self.classifier_name, self.class_weight = classifier, class_weight
        self.vocabulary = KMeans(n_clusters=n_words, random_state=seed, n_init=10, algorithm='lloyd')
        if classifier == 'svm':
            self.classifier = LinearSVC(C=C, random_state=seed, class_weight=class_weight, max_iter=10000)
        elif classifier == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=200, random_state=seed,
                                                     class_weight=class_weight, n_jobs=1)
        else:
            self.classifier = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=seed)
        self.extractor_config = {}
        self.metrics = {}

    def histogram(self, descriptors):
        """Подсчитывает и нормализует частоты визуальных слов изображения."""
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.ndim != 2 or descriptors.shape[1] != self.vocabulary.cluster_centers_.shape[1]:
            raise ValueError('Размерность дескрипторов не соответствует словарю')
        if len(descriptors) == 0:
            return np.zeros(self.n_words, dtype=np.float32)
        words = self.vocabulary.predict(descriptors)
        return np.bincount(words, minlength=self.n_words).astype(np.float32) / len(words)

    def fit(self, records, extractor, max_descriptors=20000):
        """Обучает словарь и классификатор на парах путей и классов изображений."""
        if max_descriptors < self.n_words:
            raise ValueError('Максимальное число дескрипторов должно быть не меньше числа слов')
        self.classes = sorted({label for _, label in records})
        if len(self.classes) < 2:
            raise ValueError('Для обучения необходимо как минимум два класса')
        rng = np.random.default_rng(self.seed)
        sample = np.empty((0, extractor.dimension), dtype=np.float32)
        keys = np.empty(0)
        total = 0
        started = time.perf_counter()
        for index, (path, _) in enumerate(records):
            descriptors = extractor.detect_and_compute(read_image(path))
            total += len(descriptors)
            sample = np.concatenate((sample, descriptors))
            keys = np.concatenate((keys, rng.random(len(descriptors))))
            if len(keys) > max_descriptors:
                keep = np.argpartition(keys, max_descriptors - 1)[:max_descriptors]
                sample, keys = sample[keep], keys[keep]
            if (index + 1) % 100 == 0:
                logging.info('Обработано %d/%d изображений, дескрипторов: %d', index + 1, len(records), total)
        self.metrics['extraction_seconds'] = time.perf_counter() - started
        self.metrics['descriptors'] = total
        self.metrics['vocabulary_descriptors'] = len(sample)
        if len(sample) < self.n_words:
            raise ValueError('Для обучения K-Means недостаточно дескрипторов')
        started = time.perf_counter()
        self.vocabulary.fit(sample)
        self.metrics['kmeans_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        histograms = np.stack([
            self.histogram(extractor.detect_and_compute(read_image(path)))
            for path, _ in records
        ])
        self.metrics['histogram_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        self.classifier.fit(histograms, [label for _, label in records])
        self.metrics['classifier_seconds'] = time.perf_counter() - started
        self.metrics['train_images'] = len(records)
        logging.info('Этапы обучения: %s', self.metrics)
        return self

    def predict(self, descriptors):
        """Предсказывает класс изображения по его дескрипторам."""
        return np.asarray(self.classifier.predict(self.histogram(descriptors)[None])).ravel()[0]

    def save_model(self, path):
        """Сохраняет словарь, классификатор и настройки модели в файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'format_version': self.FORMAT_VERSION, 'model': self}, path)

    @classmethod
    def load_model(cls, path):
        """Загружает сохранённую модель из файла."""
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get('format_version') != cls.FORMAT_VERSION:
            raise ValueError('Неподдерживаемый формат модели')
        model = payload.get('model')
        if not isinstance(model, cls) or not model.extractor_config or not hasattr(model.vocabulary, 'cluster_centers_'):
            raise ValueError('Неполная или несовместимая модель BoW')
        if model.vocabulary.cluster_centers_.shape[0] != model.n_words:
            raise ValueError('Размер словаря не совпадает с настройками модели')
        if list(model.classifier.classes_) != model.classes:
            raise ValueError('Классы модели не совпадают с классами классификатора')
        return model
