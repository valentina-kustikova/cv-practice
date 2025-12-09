# Детальное описание алгоритмов

## 1. Изменение размера изображения
**Функция:** `resize_image(image, new_width, new_height, interpolation='nearest')`

### Входные параметры:
- `image` - исходное изображение (numpy array H×W×C)
- `new_width`, `new_height` - целевые размеры
- `interpolation` - метод интерполяции (`'nearest'` или `'bilinear'`)

### Выходные данные:
- Изображение нового размера (numpy array)

### Алгоритм работы:
**Метод ближайшего соседа:**

```python
scale_x = w / new_width
scale_y = h / new_height
for y in range(new_height):
    for x in range(new_width):
        src_x = min(int(x * scale_x), w - 1)
        src_y = min(int(y * scale_y), h - 1)
        resized[y, x] = image[src_y, src_x]
```

**Билинейная интерполяция:**

```python
src_x = x * scale_x
src_y = y * scale_y
x1, y1 = int(src_x), int(src_y)
x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
wx, wy = src_x - x1, src_y - y1

for c in range(3):
    a = image[y1, x1, c] * (1 - wx) + image[y1, x2, c] * wx
    b = image[y2, x1, c] * (1 - wx) + image[y2, x2, c] * wx
    resized[y, x, c] = a * (1 - wy) + b * wy
```
### Сложность: O(new_width × new_height × channels)

## 2. Сепия фильтр
**Функция:** ` apply_sepia(image, intensity=1.0) `

### Входные параметры:
- `image`: исходное изображение
- `intensity`: интенсивность эффекта (0.1-2.0)

### Выходные данные:
- Изображение с сепия-эффектом

### Алгоритм работы:
**Матрица преобразования:**

```python
sepia_filter = np.array([
    [0.393, 0.769, 0.189],  # Красный канал
    [0.349, 0.686, 0.168],  # Зеленый канал  
    [0.272, 0.534, 0.131]   # Синий канал
])
```

**Преобразование пикселей:**

```python
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        pixel = image[y, x].astype(np.float32)
        new_pixel = np.dot(sepia_filter, pixel)
        new_pixel = np.clip(new_pixel * intensity, 0, 255)
        result[y, x] = new_pixel
```
**Математическая основа: Линейное преобразование цветового пространства RGB для создания "винтажного" эффекта.**

## 3. Эффект виньетки
**Функция:** ` apply_vignette(image, strength=0.8)`

### Входные параметры:
- `image`: исходное изображение
- `strength`: сила затемнения (0.1-1.0)

### Выходные данные:
- Изображение с эффектом виньетки

### Алгоритм работы:
**Инициализация:**

```python
h, w = image.shape[:2]
center_x, center_y = w // 2, h // 2
max_distance = math.sqrt(center_x**2 + center_y**2)
```
**Расчет затемнения:**

```python
for y in range(h):
    for x in range(w):
        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        norm_distance = min(distance / max_distance, 1.0)
        vignette = 1 - strength * (1 - math.cos(norm_distance * math.pi / 2))
        
        for c in range(3):
            result[y, x, c] = image[y, x, c] * vignette
```
**Физическая интерпретация: Имитация оптического явления падения освещенности к краям кадра.**

## 4. Пикселизация области
**Функция:** ` pixelate_region(image, x, y, width, height, pixel_size=10)`

### Входные параметры:
- `image`: исходное изображение
- `x`, `y`: координаты области 
- `width`, `height`: размеры области
- `pixel_size`: размер блока пикселизации

### Выходные данные:
- Изображение с пикселизированной областью

### Алгоритм работы:
**Корректировка границ:**

```python
x = max(0, min(x, image.shape[1] - 1))
y = max(0, min(y, image.shape[0] - 1))
width = min(width, image.shape[1] - x)
height = min(height, image.shape[0] - y)
```
**Блочная обработка:**

```python
for block_y in range(y, y + height, pixel_size):
    for block_x in range(x, x + width, pixel_size):
        block_end_y = min(block_y + pixel_size, y + height)
        block_end_x = min(block_x + pixel_size, x + width)
        
        block = image[block_y:block_end_y, block_x:block_end_x]
        if block.size > 0:
            avg_color = np.mean(block, axis=(0, 1))
            result[block_y:block_end_y, block_x:block_end_x] = avg_color
```

## 5. Простая рамка
**Функция:** ` add_border(image, border_width, color=(0, 0, 0))`

### Входные параметры:
- `image`: исходное изображение
- `border_width`: толщина рамки
- `color`: цвет рамки (BGR)

### Выходные данные:
- Изображение с рамкой

### Алгоритм работы:
```python
h, w = image.shape[:2]
new_h = h + 2 * border_width
new_w = w + 2 * border_width

bordered = np.full((new_h, new_w, image.shape[2]), color, dtype=image.dtype)
bordered[border_width:border_width+h, border_width:border_width+w] = image
```
## 6. Эффект бликов (текстурный)
**Функция:** ` add_lens_flare_texture(image, position, texture_path=None, intensity=0.7)`

## Входные параметры:
- `image`: исходное изображение
- `position`: координаты центра блика (x, y)
- `texture_path`: путь к файлу текстуры блика (если None, используется текстура по умолчанию)
- `intensity`: интенсивность наложения текстуры

## Выходные данные:
- Изображение с эффектом блика

## Алгоритм работы:
**Загрузка текстуры:**

```python
flare_texture = cv2.imread(actual_texture_path, cv2.IMREAD_UNCHANGED)
```

**Обработка альфа-канала:**

- Если текстура не имеет альфа-канала, преобразуем в оттенки серого и используем как альфа-канал.

**Размещение текстуры:**

```python
flare_h, flare_w = flare_texture.shape[:2]
flare_x, flare_y = position
```
**Наложение текстуры с учетом альфа-канала:**

```python
for y in range(max(0, flare_y - flare_h//2), min(h, flare_y + flare_h//2)):
    for x in range(max(0, flare_x - flare_w//2), min(w, flare_x + flare_w//2)):
        tex_y = y - (flare_y - flare_h//2)
        tex_x = x - (flare_x - flare_w//2)
        
        if 0 <= tex_y < flare_h and 0 <= tex_x < flare_w:
            tex_color = flare_texture[tex_y, tex_x, :3]
            tex_alpha = flare_texture[tex_y, tex_x, 3] * intensity

            for c in range(3):
                result[y, x, c] = min(255, result[y, x, c] * (1 - tex_alpha) + tex_color[c] * tex_alpha)
```
## 7. Фигурная рамка (текстурная)
**Функция:** ` add_fancy_border_texture(image, border_width, texture_path=None)`

### Входные параметры:
- `image`: исходное изображение
- `border_width`: толщина рамки (в текущей реализации не используется явно, но задает размер рамки в текстуре)
- `texture_path`: путь к файлу текстуры рамки (если None, используется текстура по умолчанию)

### Выходные данные:
- Изображение с наложенной текстурой рамки

### Алгоритм работы:
**Загрузка текстуры:**

```python
border_texture = cv2.imread(actual_texture_path, cv2.IMREAD_UNCHANGED)
```

**Обработка альфа-канала:**

- Если текстура не имеет альфа-канала, создается простая маска, которая определяет область рамки.

**Изменение размера текстуры:**

```python
texture_color_resized = cv2.resize(texture_color, (w, h))
texture_alpha_resized = cv2.resize(texture_alpha, (w, h))
```

**Наложение текстуры:**

```python
for y in range(h):
    for x in range(w):
        alpha = texture_alpha_resized[y, x]
        if alpha > 0:
            for c in range(3):
                result[y, x, c] = result[y, x, c] * (1 - alpha) + texture_color_resized[y, x, c] * alpha
```

## 8. Текстура акварельной бумаги
**Функция:** ` apply_watercolor_paper_texture(image, texture_path=None, intensity=0.5)`

### Входные параметры:
- `image`: исходное изображение
- `texture_path`: путь к файлу текстуры бумаги (если None, используется текстура по умолчанию)
- `intensity`: интенсивность наложения текстуры

### Выходные данные:
- Изображение с текстурой бумаги

### Алгоритм работы:
**Загрузка текстуры:**

```python
paper_texture = cv2.imread(actual_texture_path)
gray_texture = cv2.cvtColor(paper_texture, cv2.COLOR_BGR2GRAY)
```
**Изменение размера текстуры под изображение:**

```python
if gray_texture.shape[0] != h or gray_texture.shape[1] != w:
    gray_texture = cv2.resize(gray_texture, (w, h))
```

**Нормализация текстуры:**

```python
texture_norm = gray_texture.astype(np.float32) / 255.0
```

**Наложение текстуры:**

```python
for y in range(h):
    for x in range(w):
        tex_val = texture_norm[y, x]
        blend_val = 1.0 - intensity + tex_val * intensity
        result[y, x] *= blend_val
```

## Интерактивная пикселизация
**Функция:** ` interactive_pixelation(original_image)`

Управление:
- ЛКМ: Выделение области для пикселизации
- r: Сброс к оригинальному изображению
- +/-: Изменение размера пикселя
- q: Выход из режима

**Алгоритм обработки мыши:**
```python
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
```
