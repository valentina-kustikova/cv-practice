from __future__ import annotations
from pathlib import Path
import logging
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

LOG = logging.getLogger(__name__)

class BagOfWords:
    """Визуальный словарь K-means, нормированные гистограммы и ML-классификатор."""

    def __init__(self, n_clusters=128, classifier="svm", random_state=42):
        if n_clusters < 2:
            raise ValueError("n_clusters должен быть не меньше 2")
        self.n_clusters = int(n_clusters)
        self.classifier_name = classifier.lower()
        self.random_state = random_state
        self.vocabulary = None
        self.classifier = None
        self.label_encoder = LabelEncoder()

    def build_vocabulary(self, descriptors, max_samples=200_000):
        valid = [np.asarray(d, dtype=np.float32) for d in descriptors if d is not None and len(d)]
        if not valid:
            raise ValueError("Не найдено ни одного дескриптора для построения словаря.")
        dim = valid[0].shape[1]
        valid = [d for d in valid if d.ndim == 2 and d.shape[1] == dim]
        all_desc = np.vstack(valid)
        if len(all_desc) < self.n_clusters:
            raise ValueError(
                f"Дескрипторов ({len(all_desc)}) меньше числа кластеров ({self.n_clusters}). "
                "Уменьшите --clusters."
            )
        if len(all_desc) > max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(all_desc), size=max_samples, replace=False)
            all_desc = all_desc[idx]
        self.vocabulary = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.random_state,
            batch_size=min(4096, max(256, self.n_clusters * 8)), n_init=3
        ).fit(all_desc)
        LOG.info("Словарь построен: %d визуальных слов, размерность %d", self.n_clusters, dim)

    def histogram(self, descriptors):
        if self.vocabulary is None:
            raise RuntimeError("Сначала необходимо построить или загрузить словарь.")
        hist = np.zeros(self.n_clusters, dtype=np.float32)
        if descriptors is not None and len(descriptors):
            words = self.vocabulary.predict(np.asarray(descriptors, dtype=np.float32))
            hist = np.bincount(words, minlength=self.n_clusters).astype(np.float32)
        total = float(hist.sum())
        if total:
            hist /= total
            hist = np.sqrt(hist)
            norm = float(np.linalg.norm(hist))
            if norm > 0:
                hist /= norm
        return hist

    def fit(self, histograms, labels):
        X = np.asarray(histograms, dtype=np.float32)
        y = self.label_encoder.fit_transform(np.asarray(labels))
        if self.classifier_name == "svm":
            self.classifier = LinearSVC(C=1.0, random_state=self.random_state)
        elif self.classifier_name == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                random_state=self.random_state, n_jobs=-1, min_samples_leaf=1
            )
        else:
            raise ValueError("classifier должен быть svm или random_forest")
        self.classifier.fit(X, y)

    # Совместимость со старым API.
    def fit_vocabulary(self, descriptors, max_samples=200_000):
        return self.build_vocabulary(descriptors, max_samples=max_samples)

    def fit_classifier(self, histograms, labels):
        return self.fit(histograms, labels)

    def predict(self, histogram):
        if self.classifier is None:
            raise RuntimeError("Классификатор не обучен.")
        hist = np.asarray(histogram, dtype=np.float32).reshape(1, -1)
        pred = self.classifier.predict(hist)
        return str(self.label_encoder.inverse_transform(pred)[0])

    def save_model(self, path, extractor_config=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "n_clusters": self.n_clusters, "classifier_name": self.classifier_name,
            "random_state": self.random_state, "vocabulary": self.vocabulary,
            "classifier": self.classifier, "label_encoder": self.label_encoder,
            "extractor_config": extractor_config or {},
        }, path, compress=3)
        LOG.info("Модель сохранена: %s", path)

    @classmethod
    def load_model(cls, path):
        data = joblib.load(path)
        obj = cls(data["n_clusters"], data["classifier_name"], data["random_state"])
        obj.vocabulary = data["vocabulary"]
        obj.classifier = data["classifier"]
        obj.label_encoder = data["label_encoder"]
        obj.extractor_config = data.get("extractor_config", {})
        return obj
