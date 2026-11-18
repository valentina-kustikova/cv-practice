import tempfile
import unittest
from pathlib import Path

import numpy as np

from image_classification.bow import BagOfWords
from image_classification.extractors import SIFTExtractor


class BagOfWordsSmokeTest(unittest.TestCase):
    def test_fit_encode_predict_and_serialization(self):
        rng = np.random.default_rng(7)
        extractor = SIFTExtractor()
        descriptor_sets = []
        labels = []
        for class_index, label in enumerate(("class_a", "class_b")):
            center = np.zeros(128, dtype=np.float32)
            center[class_index * 10 : class_index * 10 + 10] = 5.0
            for _ in range(8):
                descriptor_sets.append(
                    (center + rng.normal(0, 0.1, size=(20, 128))).astype(np.float32)
                )
                labels.append(label)

        model = BagOfWords(
            extractor=extractor,
            extractor_name="sift",
            n_clusters=4,
            classifier_name="svm",
            random_state=7,
        )
        model.fit(descriptor_sets, labels, max_descriptors=500)

        histogram = model.encode_descriptors(descriptor_sets[0])
        self.assertEqual(histogram.shape, (4,))
        self.assertAlmostEqual(float(histogram.sum()), 1.0, places=6)

        predictions, encoded = model.predict_descriptor_sets(descriptor_sets[:2])
        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(encoded.shape, (2, 4))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save_model(path)
            loaded = BagOfWords.load_model(path)
            loaded_predictions, _ = loaded.predict_descriptor_sets(descriptor_sets[:2])
            self.assertEqual(loaded_predictions.shape, (2,))
            self.assertEqual(loaded.n_clusters, 4)


if __name__ == "__main__":
    unittest.main()
