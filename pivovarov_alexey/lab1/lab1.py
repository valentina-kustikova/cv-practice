import cv2
import numpy as np

# Функция перевода изображения в оттенки серого
def to_grayscale(image):
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.stack((gray_image,)*3, axis=-1)

# Функция изменения размера изображения
def resize_image(image, new_width, new_height):
    h, w = image.shape[:2]
    # Создаем новое изображение с белым фоном
    resized_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
    # Изменяем размер изображения
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_image[0:resized.shape[0], 0:resized.shape[1]] = resized
    return resized_image

# Функция изменения разрешения изображения с добавлением белого фона
def resize_image_with_fill(image):
    # Получаем исходные размеры
    original_height, original_width = image.shape[:2]
    
    # Вычисляем новые размеры
    new_width = original_width // 2
    new_height = original_height // 2
    
    # Создаем новое изображение
    filled_image = np.ones((original_height, original_width, 3), dtype=np.uint8) * 255
    
    # Изменяем размер изображения
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Вставляем уменьшенное изображение
    start_y = (original_height - new_height) // 2
    start_x = (original_width - new_width) // 2
    filled_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized
    
    return filled_image

# Функция применения фотоэффекта сепии
def apply_sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = np.dot(image[..., :3], sepia_filter.T)
    sepia_image[sepia_image > 255] = 255  # Обрезаем значения до 255
    return sepia_image.astype(np.uint8)

# Функция применения фотоэффекта виньетки
def apply_vignette(image):
    rows, cols = image.shape[:2]
    X_resultant_matrix = cv2.getGaussianKernel(cols, cols/4)
    Y_resultant_matrix = cv2.getGaussianKernel(rows, rows/4)
    resultant_matrix = Y_resultant_matrix * X_resultant_matrix.T
    mask = 200 * resultant_matrix / np.linalg.norm(resultant_matrix)
    vignette_image = np.copy(image)

    for i in range(3):
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask
    
    return vignette_image.astype(np.uint8)

# Функция пикселизации заданной области
def pixelate_region(image, x, y, w, h, pixel_size):
    region = image[y:y+h, x:x+w]
    temp = cv2.resize(region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated_region = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = pixelated_region
    return image

# Функция добавления подписи на изображение
def add_text(image, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)
    thickness = 2
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

# Функция разделения изображения на зоны и применения фильтров
def apply_filters_by_zones(image):
    height, width = image.shape[:2]
    zones = 5
    zone_height = height // zones

    filtered_image = np.copy(image)

    # Оттенки серого
    filtered_image[0:zone_height] = to_grayscale(image[0:zone_height])
    add_text(filtered_image, "Grayscale", 10, zone_height - 10)

    # Изменение разрешения
    resized_zone = resize_image_with_fill(image[zone_height:2*zone_height])
    filtered_image[zone_height:2*zone_height] = resized_zone
    add_text(filtered_image, "Resized", 10, 2*zone_height - 10)

    # Сепия
    filtered_image[2*zone_height:3*zone_height] = apply_sepia(image[2*zone_height:3*zone_height])
    add_text(filtered_image, "Sepia", 10, 3*zone_height - 10)

    # Виньетка
    filtered_image[3*zone_height:4*zone_height] = apply_vignette(image[3*zone_height:4*zone_height])
    add_text(filtered_image, "Vignette", 10, 4*zone_height - 10)

    # Пикселизация
    filtered_image[4*zone_height:] = pixelate_region(image[4*zone_height:], 0, 0, width, zone_height, 10)
    add_text(filtered_image, "Pixelated", 10, height - 10)

    return filtered_image

def main():
    # Загрузка изображения
    image = cv2.imread('input.jpg')
    new_width = 400
    new_height = image.shape[0] * new_width // image.shape[1]
    image = resize_image(image, new_width, new_height)
    tmp = image.copy()

    # Применение фильтров к зонам
    filtered_image = apply_filters_by_zones(tmp)

    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

