import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from image_classification.bow import BagOfWords
from image_classification.extractors import FeatureExtractor


class CnnExtractorTest(unittest.TestCase):
    image = np.random.default_rng(3).integers(256, size=(64, 80, 3), dtype=np.uint8)

    def test_window_and_layer_shapes(self):
        for name, kwargs, shape in (
            ("resnet18", {}, (49, 512)),
            ("resnet50", {"layer": 6, "window_size": 64}, (16, 1024)),
            ("mobilenet_v3_small", {"window_size": 64}, (4, 576)),
            ("dinov2", {"layer": 2, "window_size": 56}, (16, 768)),
        ):
            with self.subTest(name=name):
                extractor = FeatureExtractor.create(name, device="cpu", pretrained=False, **kwargs)
                self.assertEqual(extractor.detect_and_compute(self.image).shape, shape)

    def test_nested_batchnorm_output_precedes_inplace_relu(self):
        kwargs = dict(device="cpu", pretrained=False, window_size=64)
        default = FeatureExtractor.create("resnet18", **kwargs)
        index = default.layer_names.index("layer1.0.bn1")
        extractor = FeatureExtractor.create("resnet18", layer=index, **kwargs)
        tensor = extractor.transform(self.image[:, :, ::-1].copy()).unsqueeze(0)
        model = extractor.model
        with extractor.torch.inference_mode():
            stem = model.maxpool(model.relu(model.bn1(model.conv1(tensor))))
            expected = model.layer1[0].bn1(model.layer1[0].conv1(stem))
        descriptors = extractor.detect_and_compute(self.image)
        np.testing.assert_array_equal(descriptors, expected[0].permute(1, 2, 0).reshape(-1, 64).numpy())

    def test_load_original_sequential_resnet(self):
        extractor = FeatureExtractor.create("resnet18", device="cpu", pretrained=False)
        backbone = extractor.torch.nn.Sequential(*list(extractor.model.children())[:-2])
        tensor = extractor.transform(self.image[:, :, ::-1].copy()).unsqueeze(0)
        with extractor.torch.inference_mode():
            expected = backbone(tensor)[0].permute(1, 2, 0).reshape(-1, 512).numpy()
        payload = dict(
            format_version=1, extractor_name="resnet18", extractor_state=backbone.state_dict(),
            n_clusters=2, classifier_name="svm", random_state=42, kmeans=None, classifier=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.joblib"
            joblib.dump(payload, path)
            loaded = BagOfWords.load_model(path, device="cpu")
        np.testing.assert_array_equal(loaded.extractor.detect_and_compute(self.image), expected)
