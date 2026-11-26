import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from image_classification.data_loader import ImageRecord
from image_classification.extractors import SIFTExtractor
from image_classification.pipelines import _load_descriptors
from image_classification.utils import read_image


class ImageLoadingTest(unittest.TestCase):
    def test_empty_image_is_skipped_and_next_image_is_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_image = root / "empty.jpg"
            empty_image.touch()
            valid_image = root / "valid.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((16, 16, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            valid_image.write_bytes(buffer.tobytes())

            empty_record = ImageRecord(empty_image, "empty", "test")
            valid_record = ImageRecord(valid_image, "valid", "test")
            with self.assertLogs("image_classification.pipelines", level="WARNING") as log:
                records, descriptors = _load_descriptors([empty_record, valid_record], SIFTExtractor())

        self.assertEqual(records, [valid_record])
        self.assertEqual(len(descriptors), 1)
        self.assertIn(str(empty_image), log.output[0])
        self.assertIn("Image file is empty", log.output[0])

    def test_opencv_decode_exception_includes_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "bad.jpg"
            image_path.write_bytes(b"invalid image")
            with patch("image_classification.utils.cv2.imdecode", side_effect=cv2.error("decoder failed")):
                with self.assertRaises(ValueError) as caught:
                    read_image(image_path)

        self.assertIn(str(image_path), str(caught.exception))
        self.assertIn("decoder failed", str(caught.exception))
