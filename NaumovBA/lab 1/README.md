Детальное описание алгоритмов
1. Изменение размера изображения
Функция: resize_image(image, new_width, new_height, interpolation='nearest')

Входные параметры:
image: исходное изображение (numpy array H×W×C)

new_width, new_height: целевые размеры

interpolation: метод интерполяции ('nearest' или 'bilinear')

Выходные данные:
Изображение нового размера (numpy array)

Алгоритм работы:
Метод ближайшего соседа:

python
scale_x = w / new_width
scale_y = h / new_height
for y in range(new_height):
    for x in range(new_width):
        src_x = min(int(x * scale_x), w - 1)
        src_y = min(int(y * scale_y), h - 1)
        resized[y, x] = image[src_y, src_x]
Билинейная интерполяция:

python
src_x = x * scale_x
src_y = y * scale_y
x1, y1 = int(src_x), int(src_y)
x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
wx, wy = src_x - x1, src_y - y1

for c in range(3):
    a = image[y1, x1, c] * (1 - wx) + image[y1, x2, c] * wx
    b = image[y2, x1, c] * (1 - wx) + image[y2, x2, c] * wx
    resized[y, x, c] = a * (1 - wy) + b * wy
Сложность: O(new_width × new_height × channels)

2. Сепия фильтр
Функция: apply_sepia(image, intensity=1.0)

Входные параметры:
image: исходное изображение

intensity: интенсивность эффекта (0.1-2.0)

Выходные данные:
Изображение с сепия-эффектом

Алгоритм работы:
Матрица преобразования:

python
sepia_filter = np.array([
    [0.393, 0.769, 0.189],  # Красный канал
    [0.349, 0.686, 0.168],  # Зеленый канал  
    [0.272, 0.534, 0.131]   # Синий канал
])
Преобразование пикселей:

python
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        pixel = image[y, x].astype(np.float32)
        new_pixel = np.dot(sepia_filter, pixel)
        new_pixel = np.clip(new_pixel * intensity, 0, 255)
        result[y, x] = new_pixel
Математическая основа: Линейное преобразование цветового пространства RGB для создания "винтажного" эффекта.

3. Эффект виньетки
Функция: apply_vignette(image, strength=0.8)

Входные параметры:
image: исходное изображение

strength: сила затемнения (0.1-1.0)

Выходные данные:
Изображение с эффектом виньетки

Алгоритм работы:
Инициализация:

python
h, w = image.shape[:2]
center_x, center_y = w // 2, h // 2
max_distance = math.sqrt(center_x**2 + center_y**2)
Расчет затемнения:

python
for y in range(h):
    for x in range(w):
        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        norm_distance = min(distance / max_distance, 1.0)
        vignette = 1 - strength * (1 - math.cos(norm_distance * math.pi / 2))
        
        for c in range(3):
            result[y, x, c] = image[y, x, c] * vignette
Физическая интерпретация: Имитация оптического явления падения освещенности к краям кадра.

4. Пикселизация области
Функция: pixelate_region(image, x, y, width, height, pixel_size=10)

Входные параметры:
image: исходное изображение

x, y: координаты области

width, height: размеры области

pixel_size: размер блока пикселизации

Выходные данные:
Изображение с пикселизированной областью

Алгоритм работы:
Корректировка границ:

python
x = max(0, min(x, image.shape[1] - 1))
y = max(0, min(y, image.shape[0] - 1))
width = min(width, image.shape[1] - x)
height = min(height, image.shape[0] - y)
Блочная обработка:

python
for block_y in range(y, y + height, pixel_size):
    for block_x in range(x, x + width, pixel_size):
        block_end_y = min(block_y + pixel_size, y + height)
        block_end_x = min(block_x + pixel_size, x + width)
        
        block = image[block_y:block_end_y, block_x:block_end_x]
        if block.size > 0:
            avg_color = np.mean(block, axis=(0, 1))
            result[block_y:block_end_y, block_x:block_end_x] = avg_color
5. Простая рамка
Функция: add_border(image, border_width, color=(0, 0, 0))

Входные параметры:
image: исходное изображение

border_width: толщина рамки

color: цвет рамки (BGR)

Выходные данные:
Изображение с рамкой

Алгоритм работы:
python
h, w = image.shape[:2]
new_h = h + 2 * border_width
new_w = w + 2 * border_width

bordered = np.full((new_h, new_w, image.shape[2]), color, dtype=image.dtype)
bordered[border_width:border_width+h, border_width:border_width+w] = image
6. Фигурная рамка
Функция: add_fancy_border(image, border_width=20, color=(0, 0, 0))

Входные параметры:
image: исходное изображение

border_width: толщина рамки

color: цвет рамки

Выходные данные:
Изображение с волнистой рамкой

Алгоритм работы:
Параметры волны:

python
amplitude = border_width // 3
frequency = 0.1
Генерация маски:

python
# Верхняя и нижняя границы
for x in range(new_w):
    wave_offset_top = int(amplitude * math.sin(x * frequency))
    y_top = border_width + wave_offset_top
    mask[:y_top, x] = 1
    
    wave_offset_bottom = int(amplitude * math.sin(x * frequency + math.pi))
    y_bottom = h + border_width - wave_offset_bottom
    mask[y_bottom:, x] = 1

# Левая и правая границы
for y in range(new_h):
    wave_offset_left = int(amplitude * math.sin(y * frequency))
    x_left = border_width + wave_offset_left
    mask[y, :x_left] = 1
    
    wave_offset_right = int(amplitude * math.sin(y * frequency + math.pi))
    x_right = w + border_width - wave_offset_right
    mask[y, x_right:] = 1
Применение маски:

python
for y in range(new_h):
    for x in range(new_w):
        if mask[y, x] == 1:
            bordered[y, x] = color
        else:
            src_y = y - border_width
            src_x = x - border_width
            if 0 <= src_y < h and 0 <= src_x < w:
                bordered[y, x] = image[src_y, src_x]
7. Эффект бликов
Функция: add_lens_flare(image, position, size=50, intensity=0.7)

Входные параметры:
image: исходное изображение

position: координаты центра (x, y)

size: размер области

intensity: интенсивность

Выходные данные:
Изображение с эффектом бликов

Алгоритм работы:
Система бликов:

python
flares = [
    (size, intensity, position),
    (size*0.7, intensity*0.6, (position[0]-size//2, position[1]-size//2)),
    (size*0.4, intensity*0.4, (position[0]+size//3, position[1]+size//3))
]
Гауссово распределение:

python
for flare_size, flare_intensity, flare_pos in flares:
    flare_x = min(max(flare_pos[0], 0), w-1)
    flare_y = min(max(flare_pos[1], 0), h-1)
    
    for y in range(h):
        for x in range(w):
            distance = math.sqrt((x - flare_x)**2 + (y - flare_y)**2)
            if distance < flare_size:
                flare_value = math.exp(-(distance**2) / (2*(flare_size/3)**2))
                flare_value *= flare_intensity
                
                for c in range(3):
                    result[y, x, c] = min(255, result[y, x, c] + flare_value * 255)
8. Текстура акварельной бумаги
Функция: apply_watercolor_paper(image, texture_intensity=0.3)

Входные параметры:
image: исходное изображение

texture_intensity: интенсивность текстуры

Выходные данные:
Изображение с текстурой бумаги

Алгоритм работы:
Генерация текстуры:

python
def generate_paper_texture(height, width):
    texture = np.zeros((height, width))
    
    for octave in range(4):
        scale = 2 ** octave
        amplitude = 1.0 / (scale + 1)
        
        for y in range(height):
            for x in range(width):
                value = math.sin(x * 0.01 * scale + y * 0.01 * scale) * 0.5 + 0.5
                value += math.sin(x * 0.03 * scale) * math.sin(y * 0.03 * scale)
                texture[y, x] += value * amplitude
    
    return (texture - texture.min()) / (texture.max() - texture.min())
Наложение текстуры:

python
for y in range(h):
    for x in range(w):
        texture_value = paper_texture[y, x]
        for c in range(3):
            blended = result[y, x, c] * (1 - texture_intensity) + \
                     result[y, x, c] * texture_value * texture_intensity
            result[y, x, c] = blended
Интерактивная пикселизация
Функция: interactive_pixelation(original_image)

Управление:
ЛКМ: Выделение области для пикселизации

r: Сброс к оригинальному изображению

+/-: Изменение размера пикселя

q: Выход из режима

Алгоритм обработки мыши:
python
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            # Отрисовка прямоугольника выделения
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        width, height = x2 - x1, y2 - y1
        
        if width > 0 and height > 0:
            # Применение пикселизации