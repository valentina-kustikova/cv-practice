"""Небольшие автоматические проверки фильтров (запуск: python -m unittest)."""

import unittest
import numpy as np

from image_filters import ImageFilter


class FilterTests(unittest.TestCase):
    def setUp(self):
        # Небольшое цветное изображение 12x16, не требующее файлов на диске.
        y, x = np.mgrid[0:12, 0:16]
        self.image = np.stack(
            ((x * 13) % 256, (y * 19) % 256, ((x + y) * 11) % 256), axis=-1
        ).astype(np.uint8)

    def test_resize_shape(self):
        result = ImageFilter.get_filter("resize", width=8, height=6).apply_filter(self.image)
        self.assertEqual(result.shape, (6, 8, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_grayscale_shape(self):
        result = ImageFilter.get_filter("grayscale").apply_filter(self.image)
        self.assertEqual(result.shape, (12, 16))
        self.assertEqual(result.dtype, np.uint8)

    def test_all_color_filters_return_uint8(self):
        configs = [
            ("antique", {"strength": 0.7}),
            ("fade", {"strength": 0.5}),
            ("film", {"strength": 0.8}),
            ("matte", {"strength": 1.0, "softness": 0.3}),
            ("scratches", {"noise_strength": 5, "scratch_count": 5, "seed": 7}),
            ("neon", {"strength": 0.8, "color": "cyan"}),
        ]
        for name, params in configs:
            with self.subTest(filter_name=name):
                result = ImageFilter.get_filter(name, **params).apply_filter(self.image)
                self.assertEqual(result.shape, self.image.shape)
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(np.all(result >= 0) and np.all(result <= 255))


if __name__ == "__main__":
    unittest.main()
