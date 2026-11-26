from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .extractors import FeatureExtractor, LAYER_EXTRACTORS, NEURAL_EXTRACTORS, ResNetExtractorBase


class BagOfWords:
    """Bag-of-Visual-Words model: K-means vocabulary + image classifier."""

    MODEL_FORMAT_VERSION = 2

    def __init__(
        self,
        extractor: FeatureExtractor,
        extractor_name: str,
        n_clusters: int = 128,
        classifier_name: str = "svm",
        random_state: int = 42,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        self.extractor = extractor
        self.extractor_name = extractor_name
        self.n_clusters = int(n_clusters)
        self.classifier_name = classifier_name
        self.random_state = int(random_state)
        self.kmeans: KMeans | None = None
        self.classifier = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _as_descriptor_matrix(descriptors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(descriptors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Descriptors must be a 2D matrix, got shape {matrix.shape}")
        if matrix.shape[0] == 0:
            return matrix
        if not np.isfinite(matrix).all():
            raise ValueError("Descriptors contain NaN or infinite values")
        return matrix

    def _make_classifier(self):
        if self.classifier_name == "svm":
            return SVC(C=10.0, kernel="rbf", gamma="scale", decision_function_shape="ovr")
        if self.classifier_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            )
        if self.classifier_name == "knn":
            return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        raise ValueError("classifier_name must be one of: svm, random_forest, knn")

    def build_vocabulary(
        self,
        descriptor_sets: Iterable[np.ndarray],
        max_descriptors: int = 100_000,
    ) -> None:
        """Fit the visual vocabulary using K-means over local descriptors."""
        valid_sets = [self._as_descriptor_matrix(item) for item in descriptor_sets]
        valid_sets = [item for item in valid_sets if item.shape[0] > 0]
        if not valid_sets:
            raise ValueError("No local descriptors were extracted; cannot build the visual vocabulary")

        all_descriptors = np.vstack(valid_sets)
        if max_descriptors > 0 and all_descriptors.shape[0] > max_descriptors:
            rng = np.random.default_rng(self.random_state)
            selected = rng.choice(all_descriptors.shape[0], size=max_descriptors, replace=False)
            all_descriptors = all_descriptors[selected]

        effective_clusters = min(self.n_clusters, all_descriptors.shape[0])
        if effective_clusters < 2:
            raise ValueError("At least two local descriptors are required to build a vocabulary")
        if effective_clusters != self.n_clusters:
            self.n_clusters = effective_clusters

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=100,
            algorithm="lloyd",
        )
        self.kmeans.fit(all_descriptors)

    def _require_vocabulary(self) -> KMeans:
        if self.kmeans is None:
            raise RuntimeError("The visual vocabulary has not been trained")
        return self.kmeans

    def encode_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert one descriptor set to an L1-normalized BoW histogram."""
        kmeans = self._require_vocabulary()
        matrix = self._as_descriptor_matrix(descriptors)
        if matrix.shape[0] == 0:
            return np.zeros(self.n_clusters, dtype=np.float32)
        assignments = kmeans.predict(matrix)
        histogram = np.bincount(assignments, minlength=self.n_clusters).astype(np.float32)
        total = float(histogram.sum())
        if total > 0:
            histogram /= total
        return histogram

    def transform_descriptors(self, descriptor_sets: Iterable[np.ndarray]) -> np.ndarray:
        return np.vstack([self.encode_descriptors(item) for item in descriptor_sets])

    def fit(
        self,
        descriptor_sets: Iterable[np.ndarray],
        labels: Iterable[str],
        max_descriptors: int = 100_000,
    ) -> "BagOfWords":
        descriptor_sets = list(descriptor_sets)
        labels_array = np.asarray(list(labels), dtype=str)
        if len(descriptor_sets) != len(labels_array):
            raise ValueError("The number of descriptor sets and labels must be equal")
        if len(labels_array) < 2:
            raise ValueError("At least two training images are required")
        if np.unique(labels_array).size < 2:
            raise ValueError("Training data must contain at least two classes")

        self.build_vocabulary(descriptor_sets, max_descriptors=max_descriptors)
        histograms = self.transform_descriptors(descriptor_sets)
        self.classifier = self._make_classifier()
        self.classifier.fit(histograms, labels_array)
        self.classes_ = np.unique(labels_array)
        return self

    def histogram(self, image: np.ndarray) -> np.ndarray:
        descriptors = self.extractor.detect_and_compute(image)
        return self.encode_descriptors(descriptors)

    def predict(self, image: np.ndarray) -> str:
        label, _histogram = self.predict_with_histogram(image)
        return label

    def predict_with_histogram(self, image: np.ndarray) -> tuple[str, np.ndarray]:
        if self.classifier is None:
            raise RuntimeError("The classifier has not been trained or loaded")
        histogram = self.histogram(image)
        prediction = self.classifier.predict(histogram.reshape(1, -1))[0]
        return str(prediction), histogram

    def predict_descriptor_sets(self, descriptor_sets: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.classifier is None:
            raise RuntimeError("The classifier has not been trained or loaded")
        histograms = self.transform_descriptors(descriptor_sets)
        predictions = self.classifier.predict(histograms)
        return np.asarray(predictions, dtype=str), histograms

    def save_model(self, path: str | Path) -> Path:
        if self.kmeans is None or self.classifier is None:
            raise RuntimeError("Cannot save an untrained model")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_cnn_extractor = self.extractor_name in NEURAL_EXTRACTORS
        pretrained = bool(getattr(self.extractor, "pretrained", True))
        extractor_config = {}
        extractor_state = None
        if is_cnn_extractor:
            extractor_config["window_size"] = self.extractor.window_size
            if self.extractor_name in LAYER_EXTRACTORS:
                extractor_config["layer"] = self.extractor.layer
            # Persist the CNN weights (including ImageNet weights) so the saved
            # classifier remains self-contained and does not depend on a later
            # download or an exact torchvision default-weight version.
            extractor_state = {
                name: tensor.detach().cpu()
                for name, tensor in self.extractor.model.state_dict().items()
            }

        payload = {
            "format_version": self.MODEL_FORMAT_VERSION,
            "extractor_name": self.extractor_name,
            "extractor_config": extractor_config,
            "pretrained": pretrained,
            "extractor_state": extractor_state,
            "n_clusters": self.n_clusters,
            "classifier_name": self.classifier_name,
            "random_state": self.random_state,
            "kmeans": self.kmeans,
            "classifier": self.classifier,
            "classes": self.classes_,
        }
        joblib.dump(payload, output_path, compress=3)
        return output_path

    @classmethod
    def load_model(
        cls,
        path: str | Path,
        device: str = "auto",
    ) -> "BagOfWords":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        payload = joblib.load(model_path)
        version = payload.get("format_version")
        if version not in (1, cls.MODEL_FORMAT_VERSION):
            raise ValueError("Unsupported or outdated model file format")

        extractor_name = payload["extractor_name"]
        extractor_kwargs = dict(payload.get("extractor_config", {}))
        if extractor_name in NEURAL_EXTRACTORS:
            extractor_kwargs["device"] = device
            # The complete state dict is embedded in the model file. Build the
            # architecture without downloading weights, then restore that state.
            extractor_kwargs["pretrained"] = False if payload.get("extractor_state") is not None else bool(
                payload.get("pretrained", True)
            )
        extractor = FeatureExtractor.create(extractor_name, **extractor_kwargs)

        model = cls(
            extractor=extractor,
            extractor_name=extractor_name,
            n_clusters=payload["n_clusters"],
            classifier_name=payload["classifier_name"],
            random_state=payload["random_state"],
        )
        model.kmeans = payload["kmeans"]
        model.classifier = payload["classifier"]
        state = payload.get("extractor_state")
        if state is not None:
            if version == 1 and isinstance(extractor, ResNetExtractorBase):
                # V1 numbered the backbone modules in a Sequential and omitted fc.
                restored = {}
                for key, value in state.items():
                    index, suffix = key.split(".", 1)
                    restored[f"{extractor.STAGES[int(index)]}.{suffix}"] = value
                restored.update({f"fc.{key}": value for key, value in extractor.model.fc.state_dict().items()})
                state = restored
            extractor.model.load_state_dict(state)
        model.classes_ = payload.get("classes")
        return model
