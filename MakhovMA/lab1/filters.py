import cv2
import numpy as np
import math
import random


def resize_image(image, scale):
    """Изменение размера изображения"""
    h, w = image.shape[:2]
    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    x_ratio = w / new_width if new_width > 0 else 0
    y_ratio = h / new_height if new_height > 0 else 0

    for i in range(new_height):
        for j in range(new_width):
            px = min(int(j * x_ratio), w - 1)
            py = min(int(i * y_ratio), h - 1)
            resized[i, j] = image[py, px]

    return resized


def apply_sepia(image):
    """Применение эффекта сепии к изображению в BGR"""
    # Матрица преобразования, оптимизированная для BGR
    kernel = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]])
    sepia_img = cv2.transform(image, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


def apply_vignette(image, strength=0.8):
    """Применение эффекта виньетки"""
    h, w = image.shape[:2]

    # Создаем маску виньетки
    kernel_x = cv2.getGaussianKernel(w, w / 3)
    kernel_y = cv2.getGaussianKernel(h, h / 3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()

    # Применяем маску к каждому каналу
    result = image.copy().astype(float)
    for channel in range(3):
        result[:, :, channel] = result[:, :, channel] * (mask * strength + (1 - strength))

    return np.clip(result, 0, 255).astype(np.uint8)


class PixelateSelector:
    """Класс для интерактивного выбора области пикселизации"""

    def __init__(self, image, pixel_size=10):
        self.image = image.copy()
        self.clone = image.copy()
        self.pixel_size = pixel_size
        self.roi = None
        self.drawing = False
        self.ix, self.iy = -1, -1

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.clone = self.image.copy()
                cv2.rectangle(self.clone, (self.ix, self.iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi = (min(self.ix, x), min(self.iy, y),
                        abs(self.ix - x), abs(self.iy - y))
            cv2.rectangle(self.clone, (self.ix, self.iy), (x, y), (0, 255, 0), 2)

    def select_region(self):
        """Интерактивный выбор области и применение пикселизации"""
        cv2.namedWindow('Select Region for Pixelation')
        cv2.setMouseCallback('Select Region for Pixelation', self._mouse_callback)

        while True:
            cv2.imshow('Select Region for Pixelation', self.clone)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter - применить
                break
            elif key == 27:  # ESC - отмена
                self.roi = None
                break

        cv2.destroyWindow('Select Region for Pixelation')

        if self.roi is None:
            return self.image

        # Применяем пикселизацию к выбранной области
        x, y, w, h = self.roi
        result = self.image.copy()

        for i in range(y, y + h, self.pixel_size):
            for j in range(x, x + w, self.pixel_size):
                block = self.image[i:min(i + self.pixel_size, y + h),
                        j:min(j + self.pixel_size, x + w)]
                if block.size > 0:
                    avg_color = block.mean(axis=(0, 1))
                    result[i:min(i + self.pixel_size, y + h),
                    j:min(j + self.pixel_size, x + w)] = avg_color

        return result


def add_frame(image, color, thickness, texture_type='simple', texture_path=None, frame_style='wave'):
    """
    Добавление рамки к изображению

    Args:
        image: исходное изображение
        color: цвет рамки (для simple)
        thickness: толщина рамки
        texture_type: 'simple' - простая рамка, 'fancy' - фигурная рамка с текстурой
        texture_path: путь к текстуре (обязателен для fancy)
        frame_style: стиль фигурной рамки ('wave', 'zigzag')
    """
    h, w = image.shape[:2]
    framed = image.copy()

    if texture_type == 'simple':
        # Простая прямоугольная рамка
        framed[0:thickness, :] = color
        framed[h - thickness:h, :] = color
        framed[:, 0:thickness] = color
        framed[:, w - thickness:w] = color

    elif texture_type == 'fancy':
        if texture_path is None:
            print("Ошибка: для fancy рамки необходимо указать texture_path")
            return image

        # Загружаем текстуру
        texture = cv2.imread(texture_path)
        if texture is None:
            print(f"Ошибка: не удалось загрузить текстуру из {texture_path}")
            return image

        tex_h, tex_w = texture.shape[:2]

        if frame_style == 'wave':
            # Волнистая рамка
            for i in range(thickness):
                amplitude = thickness // 2
                shift_top = int(amplitude * math.sin(8 * math.pi * i / thickness))
                shift_bottom = int(amplitude * math.cos(8 * math.pi * i / thickness))

                for j in range(w):
                    # Верхняя граница
                    if amplitude + shift_top <= j < w - amplitude + shift_top:
                        tex_i = i % tex_h
                        tex_j = j % tex_w
                        framed[i, j] = texture[tex_i, tex_j]

                    # Нижняя граница
                    if amplitude + shift_bottom <= j < w - amplitude + shift_bottom:
                        tex_i = i % tex_h
                        tex_j = j % tex_w
                        framed[h - i - 1, j] = texture[tex_i, tex_j]

            # Левая и правая волнистые границы
            for j in range(thickness):
                amplitude = thickness // 2
                shift_left = int(amplitude * math.sin(8 * math.pi * j / thickness))
                shift_right = int(amplitude * math.cos(8 * math.pi * j / thickness))

                for i in range(h):
                    # Левая граница
                    if amplitude + shift_left <= i < h - amplitude + shift_left:
                        tex_i = i % tex_h
                        tex_j = j % tex_w
                        framed[i, j] = texture[tex_i, tex_j]

                    # Правая граница
                    if amplitude + shift_right <= i < h - amplitude + shift_right:
                        tex_i = i % tex_h
                        tex_j = j % tex_w
                        framed[i, w - j - 1] = texture[tex_i, tex_j]

        elif frame_style == 'zigzag':
            # Зигзагообразная рамка
            for i in range(thickness):
                for j in range(w):
                    if (j // 20) % 2 == 0 and i < thickness * (1 + 0.5 * math.sin(j * 0.3)):
                        tex_i = i % tex_h
                        tex_j = j % tex_w
                        framed[i, j] = texture[tex_i, tex_j]
                        framed[h - i - 1, j] = texture[tex_i, tex_j]

            for j in range(thickness):
                for i in range(h):
                    if (i // 20) % 2 == 0 and j < thickness * (1 + 0.5 * math.sin(i * 0.3)):
                        tex_i = i % tex_h
                        tex_j = j % tex_w
                        framed[i, j] = texture[tex_i, tex_j]
                        framed[i, w - j - 1] = texture[tex_i, tex_j]

    return framed


def add_lens_flare(image, texture_path, intensity=0.7):
    """Добавление эффекта бликов с использованием текстуры"""
    # Загружаем текстуру блика
    flare_texture = cv2.imread(texture_path)
    if flare_texture is None:
        print(f"Ошибка: не удалось загрузить текстуру блика из {texture_path}")
        return image

    h, w = image.shape[:2]
    flare_h, flare_w = flare_texture.shape[:2]
    output = image.copy().astype(float)

    # Позиции для бликов (можно настроить)
    positions = [
        (w // 4, h // 4),  # Верхний левый угол
        (3 * w // 4, h // 4),  # Верхний правый угол
        (w // 2, h // 2),  # Центр
        (w // 4, 3 * h // 4),  # Нижний левый угол
        (3 * w // 4, 3 * h // 4)  # Нижний правый угол
    ]

    # Добавляем несколько бликов
    for center_x, center_y in positions[:3]:  # Используем 3 блика
        for i in range(flare_h):
            for j in range(flare_w):
                img_i = center_y - flare_h // 2 + i
                img_j = center_x - flare_w // 2 + j

                if 0 <= img_i < h and 0 <= img_j < w:
                    flare_val = flare_texture[i, j].astype(float)
                    # Аддитивное смешивание для эффекта свечения
                    blended = output[img_i, img_j] + flare_val * intensity
                    output[img_i, img_j] = np.clip(blended, 0, 255)

    return output.astype(np.uint8)


def add_paper_texture(image, texture_path, intensity=0.3):
    """Наложение текстуры акварельной бумаги"""
    # Загружаем текстуру бумаги
    paper_texture = cv2.imread(texture_path)
    if paper_texture is None:
        print(f"Ошибка: не удалось загрузить текстуру бумаги из {texture_path}")
        return image

    h, w = image.shape[:2]

    # Изменяем размер текстуры под размер изображения
    paper_resized = cv2.resize(paper_texture, (w, h))

    # Конвертируем в float для вычислений
    img_float = image.astype(float)
    paper_float = paper_resized.astype(float)

    # Смешиваем изображение с текстурой
    blended = cv2.addWeighted(img_float, 1 - intensity, paper_float, intensity, 0)

    return np.clip(blended, 0, 255).astype(np.uint8)


def create_sample_textures():
    """Функция для создания образцов текстур (для тестирования)"""
    # Создаем текстуру для fancy рамки
    size = 200
    fancy_texture = np.zeros((size, size, 3), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            # Создаем узор для fancy рамки
            pattern = 150 + 70 * math.sin(i * 0.1) * math.cos(j * 0.1) + 30 * math.sin(i * 0.05 + j * 0.05)
            fancy_texture[i, j] = [
                int(200 * pattern / 255),
                int(150 * pattern / 255),
                int(100 * pattern / 255)
            ]

    cv2.imwrite('textures/fancy_frame_texture.jpg', fancy_texture)

    # Создаем текстуру блика
    flare = np.zeros((200, 200, 3), dtype=np.uint8)
    center = 100

    for i in range(200):
        for j in range(200):
            distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance < center:
                intensity = 1 - (distance / center) ** 2
                flare[i, j] = [255, 255, 200] * intensity

    cv2.imwrite('textures/flare_texture.jpg', flare)

    # Создаем текстуру бумаги
    paper = np.zeros((500, 500, 3), dtype=np.uint8)

    for i in range(500):
        for j in range(500):
            grain = random.randint(-15, 15)
            wave = 40 * math.sin(i * 0.05) * math.cos(j * 0.05)
            gray = 200 + grain + wave
            paper[i, j] = [gray, gray, gray]

    cv2.imwrite('textures/paper_texture.jpg', paper)
    print("Образцы текстур созданы в папке 'textures/'")


if __name__ == "__main__":
    # Создаем образцы текстур при прямом запуске
    import os

    if not os.path.exists('textures'):
        os.makedirs('textures')
    create_sample_textures()