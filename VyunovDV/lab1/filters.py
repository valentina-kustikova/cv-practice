import os
import cv2
import numpy as np
import argparse
from os.path import exists

parser = argparse.ArgumentParser(prog='Filters')
parser.add_argument('--image_path', help='Image path')
# parser.add_argument('--filter', type=str)

class Filters:
    def __init__(self, image_path: str) -> None:
        try:
            self.image = cv2.imread(image_path)
            # self.roi = cv2.selectROI('window', self.image, showCrosshair=True)
            # print(self.roi)
            if self.image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            self.height, self.width, self.channels = self.image.shape
        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise

    def apply_grayscale(self) -> np.ndarray:
        """Apply grayscale filter using the average of the RGB channels."""
        try:
            grayscale_img = self.image.mean(axis=2).astype(np.uint8)
            return grayscale_img
        except Exception as e:
            raise


    def apply_sepia(self) -> np.ndarray:
        """Apply sepia filter."""
        try:
            kernel = np.array(
                [[0.272, 0.534, 0.131],
                 [0.349, 0.686, 0.168],
                 [0.393, 0.769, 0.189]]
            )
            sepia_img = cv2.transform(self.image, kernel)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            return sepia_img
        except Exception as e:
            raise

    def resize_image(self, new_width: int, new_height: int) -> np.ndarray:
        """Resize image using nearest neighbor interpolation."""
        try:
            original_height, original_width = self.image.shape[:2]
        
            # Создаем сетку координат для нового изображения
            x = np.arange(int(new_width)) * (original_width / int(new_width))
            y = np.arange(int(new_height)) * (original_height / int(new_height))
            
            # Округляем координаты до ближайшего целого числа
            x = np.floor(x).astype(int)
            y = np.floor(y).astype(int)
            
            # Используем индексацию для получения нового изображения
            resized_img = self.image[y[:, None], x]
            
            return resized_img
        except Exception as e:
            raise e

    def apply_vignette(self, radius: int = 0.1) -> np.ndarray:
        """Apply vignette filter."""
        try:
            radius = max(0.1, min(radius, 2.0))
            x = np.linspace(-1, 1, self.width)
            y = np.linspace(-1, 1, self.height)
            xv, yv = np.meshgrid(x, y)
            vignette_mask = np.sqrt(xv**2 + yv**2) / radius
            vignette_mask = (1 - np.clip(vignette_mask, 0, 1)) * 255
            vignette_mask = vignette_mask.astype(np.uint8)
            vignette_img = cv2.merge([vignette_mask] * self.channels) / 255 * self.image
            return vignette_img.astype(np.uint8)
        except Exception as e:
            raise

    def pixelate_area(self, top_left: tuple[int, int], bottom_right: tuple[int, int], pixel_size: int) -> np.ndarray:
        """Pixelate a specified area."""
        try:

            pixelated_img = self.image.copy()
            x1, y1 = top_left
            x2, y2 = bottom_right
            for y in range(y1, y2, pixel_size):
                for x in range(x1, x2, pixel_size):
                    roi = pixelated_img[y:y + pixel_size, x:x + pixel_size]
                    color = roi.mean(axis=(0, 1)).astype(int)
                    pixelated_img[y:y + pixel_size, x:x + pixel_size] = color
            return pixelated_img
        except Exception as e:
            raise


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        os.makedirs("results", exist_ok=True)
        if not all(exists(i) for i in args.image_path.split(',')):
            exit(f'File {args.image_path} is not found')
        images = args.image_path.split(',')
        operations = [
            ("grayscale", lambda f: f.apply_grayscale(), "results/grayscale_{0}.jpg"),
            ("sepia", lambda f: f.apply_sepia(), "results/sepia_{0}.jpg"),
            ("resize", lambda f: f.resize_image(200, 150), "results/resized_{0}.jpg"),
            ("resize_high", lambda f: f.resize_image(1920, 1080), "results/resized_high_{0}.jpg"),
            ("vignette", lambda f: f.apply_vignette(), "results/vignette_{0}.jpg"),
            ("pixelate", lambda f: f.pixelate_area((50, 50), (200, 200), 10), "results/pixelated_{0}.jpg"),
        ]

        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            filters = Filters(img_path)
            for i in ['grayscale', 'sepia', 'resize', 'vignette', 'pixelate']:
                need = input(f'Need {i}? ')
                if int(need):
                    if i == 'vignette':
                        radius = int(input('Radius = '))
                        result = filters.apply_vignette(radius)
                        cv2.imwrite('vignette.jpg', result)
                    elif i == 'resize':
                        width, height = input('Width, height: ').split(' ')
                        result = filters.resize_image(width, height)
                        cv2.imwrite('resize.jpg', result)
                    elif i == 'pixelate':
                        image = cv2.imread(img_path)
                        roi = cv2.selectROI('window', image, showCrosshair=True)
                        pixel_size = int(input('Pixel size'))
                        top_left = roi[:2]
                        bottom_right = [roi[0] + roi[2], roi[1] + roi[3]]
                        result = filters.pixelate_area(top_left, bottom_right, pixel_size)
                        cv2.imwrite('pixelate.jpg', result)
                    elif i == 'grayscale':
                        result = filters.apply_grayscale()
                        cv2.imwrite('grayscale.jpg', result)
                    elif i == 'sepia':
                        result = filters.apply_sepia()
                        cv2.imwrite('sepia.jpg', result)
                    
            for name, func, output in operations:
                result = func(filters)
                cv2.imwrite(output.format(img_name), result)
    except Exception as e:
        raise