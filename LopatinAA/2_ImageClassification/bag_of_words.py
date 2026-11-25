
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


class BagOfWords:

    def __init__(self, extractor_name, class_names, vocab_size=300,
                 max_descriptors=150000, *, layer=None):
        if vocab_size < 2:
            raise ValueError("Размер словаря должен быть не меньше 2")
        self.extractor_name = extractor_name
        self.layer = layer
        self.class_names = list(class_names)
        self.vocab_size = vocab_size
        self.max_descriptors = max_descriptors
        self.kmeans = None      # словарь визуальных слов
        self.classifier = None  # обученный классификатор

    # ------------------------------------------------------------------
    # Словарь
    # ------------------------------------------------------------------

    def build_vocabulary(self, descriptor_list):
        """Строит словарь визуальных слов методом K-средних.
            descriptor_list: список матриц дескрипторов.
        """
        non_empty = [descriptors for descriptors in descriptor_list if len(descriptors) > 0]
        if len(non_empty) == 0:
            raise ValueError("Нет ни одного дескриптора для построения словаря")
        all_descriptors = np.vstack(non_empty)

        if len(all_descriptors) < self.vocab_size:
            raise ValueError(f"Дескрипторов ({len(all_descriptors)}) меньше, "
                             f"чем размер словаря ({self.vocab_size})")

        # Для ускорения кластеризации используется случайная подвыборка дескрипторов
        if len(all_descriptors) > self.max_descriptors:
            generator = np.random.default_rng(RANDOM_SEED)
            indices = generator.choice(len(all_descriptors), self.max_descriptors, replace=False)
            all_descriptors = all_descriptors[indices]
        all_descriptors = all_descriptors.astype(np.float32)

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            batch_size=4096, # размер выборки дескрипторов на один шаг обновления центров
            n_init=3, # число пробных начальных точек для центра кластера
            random_state=RANDOM_SEED,
        )
        self.kmeans.fit(all_descriptors)

    def compute_histogram(self, descriptors):
        """Вычисляет нормализованную BoW-гистограмму для одного изображения."""
        if self.kmeans is None:
            raise RuntimeError("Словарь не построен: вызовите build_vocabulary или load_model")

        histogram = np.zeros(self.vocab_size, dtype=np.float32)
        if len(descriptors) == 0:
            return histogram

        words = self.kmeans.predict(descriptors.astype(np.float32))
        counts = np.bincount(words, minlength=self.vocab_size)
        histogram = counts.astype(np.float32) / len(words)
        return histogram

    def compute_histograms(self, descriptor_list):
        """BoW-гистограммы для списка изображений."""
        histograms = [self.compute_histogram(descriptors) for descriptors in descriptor_list]
        return np.array(histograms, dtype=np.float32)

    # ------------------------------------------------------------------
    # Классификатор
    # ------------------------------------------------------------------

    def create_classifier(self):
        """Создает случайный лес + StandartScaler."""
        model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model),
        ])

    def train(self, histograms, labels):
        if len(histograms) != len(labels):
            raise ValueError("Число гистограмм не совпадает с числом меток")
        self.classifier = self.create_classifier()
        self.classifier.fit(histograms, labels)

    def predict(self, histograms):
        self._check_trained()
        return self.classifier.predict(histograms)

    def predict_scores(self, histograms):
        """Возвращает матрицу достоверностей."""
        self._check_trained()
        raw_scores = self.classifier.predict_proba(histograms)

        # Дополнение до классов, не встретившихся при обучении
        scores = np.full((len(histograms), len(self.class_names)), -np.inf, dtype=np.float64)
        known_classes = self.classifier.classes_
        for column, class_id in enumerate(known_classes):
            scores[:, class_id] = raw_scores[:, column]
        return scores

    def _check_trained(self):
        if self.classifier is None:
            raise RuntimeError("Классификатор не обучен: вызовите train или load_model")

    # ------------------------------------------------------------------
    # Сохранение и загрузка
    # ------------------------------------------------------------------

    def save_model(self, path):
        """Сохраняет словарь, классификатор и параметры модели в файл."""
        self._check_trained()
        data = {
            "extractor_name": self.extractor_name,
            "layer": self.layer,
            "class_names": self.class_names,
            "vocab_size": self.vocab_size,
            "max_descriptors": self.max_descriptors,
            "kmeans": self.kmeans,
            "classifier": self.classifier,
        }
        joblib.dump(data, path, compress=3)

    @staticmethod
    def load_model(path):
        """Загружает модель из файла и возвращает объект BagOfWords."""
        data = joblib.load(path)
        classifier = data["classifier"]
        if not isinstance(classifier, Pipeline) or not isinstance(
            classifier.named_steps.get("classifier"), RandomForestClassifier
        ):
            raise ValueError("Файл модели должен содержать случайный лес")
        model = BagOfWords(
            extractor_name=data["extractor_name"],
            layer=data.get("layer"),
            class_names=data["class_names"],
            vocab_size=data["vocab_size"],
            max_descriptors=data["max_descriptors"],
        )
        model.kmeans = data["kmeans"]
        model.classifier = classifier
        return model
