import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def change_resolution(image, scale_factor):
    """
    Изменяет разрешение изображения
    """
    if scale_factor <= 0:
        raise ValueError("Коэффициент масштабирования должен быть положительным")
    
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def apply_sepia(image):
    """
    Применяет фотоэффект сепии к изображению
    """
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

def apply_vignette(image, strength=0.8):
    """
    Применяет эффект виньетки к изображению
    """
    height, width = image.shape[:2]
    
    # Создаем маску виньетки
    kernel_x = cv2.getGaussianKernel(width, width/3)
    kernel_y = cv2.getGaussianKernel(height, height/3)
    kernel = kernel_y * kernel_x.T
    
    # Нормализуем и применяем силу эффекта
    mask = kernel / kernel.max()
    mask = 1 - (1 - mask) * strength
    
    # Применяем маску к каждому каналу
    vignette_image = image.copy().astype(np.float32)
    for i in range(3):
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask
    
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)
    return vignette_image

def pixelate_region(image, x, y, width, height, pixel_size=10):
    """
    Пикселизирует заданную прямоугольную область изображения
    """
    pixelated_image = image.copy()
    
    # Проверяем границы и корректируем при необходимости
    img_height, img_width = image.shape[:2]
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    if width <= 0 or height <= 0:
        return image
    
    # Вырезаем область для пикселизации
    region = image[y:y+height, x:x+width]
    
    # Уменьшаем разрешение
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)
    small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    
    # Увеличиваем обратно до исходного размера
    pixelated_region = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Вставляем обратно в изображение
    pixelated_image[y:y+height, x:x+width] = pixelated_region
    return pixelated_image

def add_solid_border(image, border_width=10, color=(0, 0, 0)):
    """
    Добавляет прямоугольную одноцветную рамку
    """
    bordered_image = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return bordered_image

def add_patterned_border(image, border_width=20, pattern_type='dashed'):
    """
    Добавляет фигурную рамку по краям изображения
    """
    height, width = image.shape[:2]
    bordered_image = image.copy()
    
    if pattern_type == 'dashed':
        # Пунктирная рамка
        dash_length = border_width * 2
        for i in range(0, width, dash_length):
            if i + dash_length <= width:
                # Верхняя граница
                bordered_image[0:border_width, i:i+dash_length//2] = [255, 0, 0]
                # Нижняя граница
                bordered_image[height-border_width:height, i:i+dash_length//2] = [255, 0, 0]
        
        for i in range(0, height, dash_length):
            if i + dash_length <= height:
                # Левая граница
                bordered_image[i:i+dash_length//2, 0:border_width] = [255, 0, 0]
                # Правая граница
                bordered_image[i:i+dash_length//2, width-border_width:width] = [255, 0, 0]
    
    elif pattern_type == 'dots':
        # Точечная рамка
        dot_spacing = border_width
        for i in range(border_width//2, width, dot_spacing):
            # Верхняя и нижняя границы
            bordered_image[border_width//2:border_width, i:i+border_width//2] = [0, 255, 0]
            bordered_image[height-border_width:height-border_width//2, i:i+border_width//2] = [0, 255, 0]
        
        for i in range(border_width//2, height, dot_spacing):
            # Левая и правая границы
            bordered_image[i:i+border_width//2, border_width//2:border_width] = [0, 255, 0]
            bordered_image[i:i+border_width//2, width-border_width:width-border_width//2] = [0, 255, 0]
    
    return bordered_image

def apply_lens_flare(image, position=None, size=50):
    """
    Накладывает эффект бликов объектива камеры
    """
    flare_image = image.copy().astype(np.float32)
    height, width = image.shape[:2]
    
    # Если позиция не указана, ставим блик в центр
    if position is None:
        position = (width // 2, height // 2)
    
    # Создаем блик
    y, x = np.ogrid[:height, :width]
    center_x, center_y = position
    
    # Создаем круговой градиент для блика
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    flare = np.exp(-distance**2 / (2 * (size**2)))
    flare = flare * 150  # Усиливаем эффект
    
    # Добавляем блик к изображению
    for i in range(3):
        flare_image[:, :, i] += flare
    
    flare_image = np.clip(flare_image, 0, 255).astype(np.uint8)
    return flare_image

def apply_watercolor_texture(image, texture_intensity=0.3):
    """
    Накладывает текстуру акварельной бумаги
    """
    height, width = image.shape[:2]
    
    # Создаем текстуру шума (имитация бумаги)
    noise = np.random.normal(0, texture_intensity * 50, (height, width))
    noise = np.stack([noise, noise, noise], axis=2)
    
    # Добавляем текстуру к изображению
    textured_image = image.astype(np.float32) + noise
    textured_image = np.clip(textured_image, 0, 255).astype(np.uint8)
    
    # Легкое размытие для акварельного эффекта
    textured_image = cv2.GaussianBlur(textured_image, (3, 3), 0)
    
    return textured_image

def main():
    # Загружаем изображение
    image_path = 'image.jpg'
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}")
        print("Убедитесь, что файл image.jpg находится в той же папке, что и скрипт")
        return

    # Конвертируем в RGB для отображения
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Получаем размеры изображения для корректной пикселизации
    img_height, img_width = original_image_rgb.shape[:2]
    
    # Автоматически рассчитываем безопасные координаты для пикселизации
    pixelate_x = img_width // 4
    pixelate_y = img_height // 4
    pixelate_width = min(200, img_width // 2)
    pixelate_height = min(150, img_height // 2)

    # Демонстрация всех фильтров
    filters = [
        ("Изменение разрешения (0.5x)", change_resolution(original_image_rgb, 0.5)),
        ("Эффект сепии", apply_sepia(original_image_rgb)),
        ("Виньетка", apply_vignette(original_image_rgb)),
        ("Пикселизация области", pixelate_region(original_image_rgb, pixelate_x, pixelate_y, pixelate_width, pixelate_height)),
        ("Прямоугольная рамка", add_solid_border(original_image_rgb, 15, (255, 0, 0))),
        ("Фигурная рамка (пунктир)", add_patterned_border(original_image_rgb, 15, 'dashed')),
        ("Блик объектива", apply_lens_flare(original_image_rgb)),
        ("Акварельная текстура", apply_watercolor_texture(original_image_rgb))
    ]

    # Создаем сетку для отображения
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    # Оригинальное изображение
    axes[0].imshow(original_image_rgb)
    axes[0].set_title('Оригинальное изображение')
    axes[0].axis('off')

    # Примененные фильтры
    for i, (title, filtered_image) in enumerate(filters, 1):
        axes[i].imshow(filtered_image)
        axes[i].set_title(title)
        axes[i].axis('off')

    # Скрываем лишние subplot'ы если они есть
    for i in range(len(filters) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()