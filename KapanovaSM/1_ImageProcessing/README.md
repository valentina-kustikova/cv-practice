# Обработка изображений с использованием OpenCV

Практическая работа №1. Обработка изображений с использованием библиотеки OpenCV.

## Структура проекта

Корневая папка — `1_ImageProcessing`. Внутри неё: пакет `filters/`
(с файлами `__init__.py`, `base.py`, `resize.py`, `grayscale.py`,
`antique.py`, `fade.py`, `film.py`, `matte.py`, `aged.py`, `neon.py`),
папка `images/` с тестовым изображением `image.jpg`, папка `textures/`
с текстурой царапин, скрипт `main.py` — точка входа, файл зависимостей
`requirements.txt` и описание `README.md`.

```
1_ImageProcessing/
    filters/
        __init__.py        — делает папку пакетом
        base.py            — абстрактный класс ImageFilter
        resize.py          — изменение разрешения
        grayscale.py       — перевод в оттенки серого
        antique.py         — эффект «антиквариат»
        fade.py            — выцветание
        film.py            — имитация плёнки
        matte.py           — овальная рамка
        aged.py            — состаренная фотография (текстура + шум)
        neon.py            — неоновый эффект
    images/
        image.jpg          — тестовое изображение
    textures/
        scratches.png      — текстура царапин
    main.py
    requirements.txt
    README.md
```

---

## Установка окружения

    conda create -n cv_lab1 python=3.11 -y
    conda activate cv_lab1
    pip install -r requirements.txt

Содержимое `requirements.txt`:

    opencv-python
    numpy

---

## Запуск

    python main.py -i <input> -o <output> -f <filter> [параметры]

| Аргумент | Описание |
|---|---|
| `-i, --input`  | путь к входному изображению |
| `-o, --output` | путь для сохранения результата |
| `-f, --filter` | тип фильтра |

### Параметры фильтров

| Параметр | Фильтр | Описание | По умолчанию |
|---|---|---|---|
| `--scale`             | resize    | масштаб (>0) | 0.5 |
| `--vignette`          | antique   | сила виньетки (>= 0) | 0.4 |
| `--antique_noise`     | antique   | уровень зерна (>= 0) | 6.0 |
| `--strength`          | fade      | сила выцветания (0..1) | 0.5 |
| `--film_strength`     | film      | сила эффекта плёнки (0..1) | 0.7 |
| `--film_noise`        | film      | уровень зерна | 8.0 |
| `--size`              | matte     | размер маски (0..1) | 0.9 |
| `--softness`          | matte     | мягкость края (>= 0) | 0.15 |
| `--texture`           | aged      | путь к PNG-текстуре | None |
| `--noise_level`       | aged      | уровень шума | 10.0 |
| `--scratch_intensity` | aged      | заметность текстуры (0..1) | 0.6 |
| `--neon_strength`     | neon      | усиление контура (>0) | 9.0 |
| `--neon_threshold`    | neon      | порог контура (>= 0) | 20.0 |

---

## Примеры запуска

### Изменение разрешения
    python main.py -i images/image.jpg -o out_resize.jpg -f resize --scale 0.5

### Оттенки серого
    python main.py -i images/image.jpg -o out_gray.jpg -f gray

### Антиквариат (сепия + виньетка + зерно)
    python main.py -i images/image.jpg -o out_antique.jpg -f antique --vignette 1.5 --antique_noise 6

### Выцветание
    python main.py -i images/image.jpg -o out_fade.jpg -f fade --strength 0.5

### Плёнка
    python main.py -i images/image.jpg -o out_film.jpg -f film --film_strength 0.5

### Овальная рамка (маска)
    python main.py -i images/image.jpg -o out_matte.jpg -f matte --size 0.9 --softness 0.15

### Состаренная фотография (текстура + шум)
    python main.py -i images/image.jpg -o out_aged.jpg -f aged --texture textures/scratches.png --noise_level 10 --scratch_intensity 0.8

### Неоновый эффект
    python main.py -i images/image.jpg -o out_neon.jpg -f neon --neon_strength 5 --neon_threshold 180
    python main.py -i images/image.jpg -o out_neon.jpg -f neon --neon_strength 5 --neon_threshold 69
    python main.py -i images/image.jpg -o out_neon.jpg -f neon --neon_strength 10 --neon_threshold 69

---

## Математическое описание фильтров

### 1. Изменение разрешения (Resize)

Метод **ближайшего соседа** с использованием linspace и округления к меньшему:

    row_idx = floor(linspace(0, h-1, new_h))
    col_idx = floor(linspace(0, w-1, new_w))

где `new_h = h * scale`, `new_w = w * scale`, `floor` — округление
к меньшему (округление вниз).

Координаты равномерно распределены по исходному изображению через
`np.linspace(0, h-1, new_h)`. Реализация — через `np.linspace` + `np.floor`
+ индексацию пикселей. 


### 2. Оттенки серого (ToGray)

Y = 0.299 * R + 0.587 * G + 0.114 * B

Формула яркости в цветовом пространстве YUV: Y — яркость, U и V — цветовая информация.
Берём только яркость и используем её как серое изображение, а U и V отбрасываем.

Результат — одноканальное изображение `Y` размером `H × W`.


### 3. Antique — сепия + виньетка + шум

#### Этап 1. Сепия

Линейное преобразование каналов через матрицу `M_sepia`:

    | R' |   | 0.393  0.769  0.189 |   | R |
    | G' | = | 0.349  0.686  0.168 | * | G |
    | B' |   | 0.272  0.534  0.131 |   | B |

Даёт тёплый коричневато-жёлтый оттенок.

#### Этап 2. Виньетка

Затемнение по краям. Нормированное расстояние от центра:

    d(x, y) = sqrt( ((x − c_x) / c_x)^2 + ((y − c_y) / c_y)^2 ) / d.max()

где `c_x = W/2`, `c_y = H/2`. В центре `d = 0`, на границе `d = 1`.

Маска затемнения:

    v(x, y) = clip( 1 − vignette_strength · d^2, 0, 1 )
    I' = I · v

Применение:

    I_sepia(x, y) ← v(x, y) · I_sepia(x, y)

#### Этап 3. Шум

К каждому пикселю добавляется независимая гауссова величина:

    I_final = I_sepia + N(0, sigma²)

где `sigma = antique_noise` 


### 4. FadeColor — выцветание

Линейное смешивание с белым:

    I' = I · (1 − s) + 255 · s

где `s = strength` ∈ [0, 1].

При `s = 0` изображение не меняется; при `s = 1` — становится полностью
белым. Промежуточные значения имитируют утрату контраста.


### 5. Film — имитация инфракрасной плёнки

#### Шаг 1. Сдвиг каналов

    R' = R · (1 + 0.6 · s)
    G' = G · (1 − 0.4 · s)
    B' = B · (1 − 0.7 · s)

где `s = strength` ∈ [0, 1].

При `s = 0` изображение не меняется; при `s = 1`:
- R усиливается в 1.6 раза,
- G падает до 60%,
- B падает до 30%.

Итог — **насыщенный красный оттенок**: R доминирует

#### Шаг 2. Шум через гауссово распределение

    I_final = clip( I_tinted + N(0, sigma²), 0, 255 )

где `sigma = noise_sigma` (по умолчанию 8.0).


### 6. Matte — мягкая овальная рамка

Нормированное расстояние до центра:

    d(x, y) = sqrt( ((x − c_x) / a)^2 + ((y − c_y) / b)^2 )

где `(c_x, c_y)` — центр изображения, `a = (W/2) · size`, `b = (H/2) · size`.

Мягкая маска через **сигмоиду**:

    alpha(x, y) = 1 / (1 + exp((d − 1) / softness))

При `softness = 0` — **жёсткий порог**: `alpha = (d <= 1)`.

Итог:

    I' = I · alpha + 255 · (1 − alpha)


### 7. Aged — состаренная фотография

#### Шаг 1. Наложение текстуры через `np.tile`

Текстура размножается по сетке:

    tex_tiled = np.tile(tex, (reps_y, reps_x, 1))
    tex = tex_tiled[:H, :W]

Тёмные места текстуры **тянут фото к заданному цвету `paper_color`**:

    mix = (1 − tex_norm) · k
    I' = I · (1 − mix) + paper_color · mix

где `k = scratch_intensity`.


#### Шаг 2. Шум

    I_final = I' + N(0, sigma²)

где `sigma = noise_level` (по умолчанию 10.0).


### 8. Neon — неоновый эффект

#### Шаг 1. Оттенки серого


    Y = 0.299 · R + 0.587 · G + 0.114 · B

#### Шаг 2. Оператор Собеля

Ядра:

    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]

Свёртка выполняется **через numpy slicing**.

#### Шаг 3. Магнитуда градиента

    |G| = sqrt(Gx² + Gy²)

Большая `|G|` — резкий перепад яркости — граница объекта.

#### Шаг 4. Усиление

    magnitude = |G| · strength

#### Шаг 5. Порог

    magnitude = where(magnitude > threshold, magnitude, 0)

где `threshold = neon_threshold` (по умолчанию 20.0).

Слабые контуры обнуляются.

#### Шаг 6. Обрезка

    magnitude = clip(magnitude, 0, 255)

#### Шаг 7. Наложение на оригинал

    B += magnitude
    G += magnitude
    (R не трогаем — получаем бирюзовый цвет)
