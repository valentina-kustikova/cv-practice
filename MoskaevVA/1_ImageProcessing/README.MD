# Практическая работа №1. Обработка изображений

## Установка и запуск

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py -i input.png -o output.png -f <filter> [параметры]

Скрипт выполняет: разбор аргументов командной строки -> чтение изображения
с обработкой исключений -> применение фильтра -> сохранение результата.
Ошибки чтения/записи/неизвестного фильтра логируются и возвращают код 1.

## Фильтры

| Фильтр    | Параметры                                                         | Описание                          |
|-----------|-------------------------------------------------------------------|-----------------------------------|
| resize    | --width, --height                                                 | билинейное изменение разрешения   |
| grayscale | —                                                                 | оттенки серого (BT.601)           |
| antique   | --strength, --grain, --seed                                       | сепия + виньетка + зерно          |
| fade      | --strength, --black-lift                                          | выцветание, сжатие диапазона      |
| film      | --mix, --grain, --seed                                            | инфракрасная плёнка + зерно       |
| matte     | --softness, --scale                                               | белая овальная рамка (smoothstep) |
| scratches | --n-scratches, --noise-sigma, --seed, --dust, --vertical-bias     | царапины, шум, пыль               |
| neon      | --threshold, --glow-radius, --glow-strength                       | неоновые контуры + свечение       |

## Примеры

# resize
python main.py -i in.png -o out/resize.png    -f resize --width 800 --height 600

# grayscale
python main.py -i in.png -o out/grayscale.png -f grayscale

# antique — с зерном
python main.py -i in.png -o out/antique.png   -f antique --strength 1.2 --grain 0.3

# fade — со сжатием диапазона
python main.py -i in.png -o out/fade.png      -f fade --strength 0.6 --black-lift 40

# film — инфракрасная плёнка
python main.py -i in.png -o out/film.png      -f film --mix 0.7 --grain 0.3

# matte — smoothstep
python main.py -i in.png -o out/matte.png     -f matte --softness 0.2 --scale 0.85

# scratches — с «мусором» и затуханием
python main.py -i in.png -o out/scratches.png -f scratches --n-scratches 12 --noise-sigma 15 --dust 0.006

# neon — с glow
python main.py -i in.png -o out/neon.png      -f neon --threshold 25 --glow-radius 3 --glow-strength 1.4

---

# Математическое описание алгоритмов

## 1. Resize — билинейная интерполяция

### Постановка

Дано изображение I размера H_s x W_s. Требуется получить изображение I'
размера H_d x W_d.

### Отображение координат

Для каждого пикселя целевого изображения (x_d, y_d) вычисляется координата
в исходном:

    x_s = (x_d + 0.5) * (W_s / W_d) - 0.5
    y_s = (y_d + 0.5) * (H_s / H_d) - 0.5

Слагаемое 0.5 нужно, чтобы центры пикселей совпадали. Координаты
ограничиваются диапазонами [0, W_s - 1] и [0, H_s - 1].

### Интерполяция

Обозначим:

    x0 = floor(x_s),  x1 = min(x0 + 1, W_s - 1)
    y0 = floor(y_s),  y1 = min(y0 + 1, H_s - 1)
    wx = x_s - x0,    wy = y_s - y0

Значение канала:

    I'(x_d, y_d) = (1 - wx)(1 - wy) * I[y0, x0]
                 +      wx (1 - wy) * I[y0, x1]
                 + (1 - wx)     wy * I[y1, x0]
                 +      wx      wy * I[y1, x1]

Эквивалентно двум линейным интерполяциям:

    top    = (1 - wx) * I[y0, x0] + wx * I[y0, x1]
    bottom = (1 - wx) * I[y1, x0] + wx * I[y1, x1]
    I'     = (1 - wy) * top       + wy * bottom

Сложность O(W_d * H_d) на канал.

## 2. Grayscale — оттенки серого (BT.601)

    Y = 0.299 * R + 0.587 * G + 0.114 * B

Коэффициенты отражают чувствительность глаза: зелёный вносит наибольший
вклад, синий — наименьший. Сумма коэффициентов равна 1, поэтому
Y принадлежит [0, 255].

Результат дублируется в три канала: I'(x, y, c) = Y(x, y) для всех c.

## 3. Antique — сепия + виньетка + зерно

### Сепия

Линейное преобразование вектора (R, G, B) матрицей 3x3:

    R' = 0.393 * R + 0.769 * G + 0.189 * B
    G' = 0.349 * R + 0.686 * G + 0.168 * B
    B' = 0.272 * R + 0.534 * G + 0.131 * B

Сумма по строкам больше 1, поэтому светлые участки выгорают
в жёлто-коричневый.

### Виньетка

    d(x, y)  = sqrt( (x - cx)^2 + (y - cy)^2 )
    d_norm   = d(x, y) / d_max
    V(x, y)  = 1 - 0.6 * d_norm * strength
    I_1      = I_sepia * V

(cx, cy) — центр изображения, d_max — максимальное расстояние (до угла).

### Зерно, зависящее от яркости

Вычисляется яркость:

    lum(x, y) = (0.299*R + 0.587*G + 0.114*B) / 255

Маска, сильнее в средних тонах и слабее в светах и тенях:

    mask(x, y) = clip( 1 - 4*(lum - 0.5)^2,  0.1,  1 )

Шум:

    eta(x, y) ~ N(0, (grain * 20)^2)
    I'(x, y, c) = I_1(x, y, c) + eta(x, y) * mask(x, y)

Seed фиксирован для воспроизводимости.

## 4. Fade — выцветание и сжатие диапазона

### Десатурация

    mean(x, y)   = (R + G + B) / 3
    I_1(x, y, c) = (1 - s) * I(x, y, c) + s * mean(x, y)

s = strength, s принадлежит [0, 1].

### Сжатие динамического диапазона

Диапазон [0, 255] сжимается в [f_min, f_max]:

    f_min = black_lift
    f_max = 255 - black_lift
    I'(x, y, c) = f_min + I_1(x, y, c) * (f_max - f_min) / 255

Если f_max <= f_min (black_lift >= 128), изображение становится
однородным серым уровня f_min.

## 5. InfraredFilm — инфракрасная плёнка + зерно

### ИК-яркость

    IR(x, y) = 1.3 * G

Листва сильно отражает ближний ИК и соответствует яркому зелёному каналу,
поэтому за оценку ИК-яркости берётся зелёный канал с усилением. Синий
используется как «холодная» составляющая — небо в ИК тёмное, его вклад
гасит зелень.

### Подмешивание

    R'(x, y) = (1 - m) * R + m * IR
    G'(x, y) = (1 - m) * G + m * B * 0.2
    B'(x, y) = (1 - m) * B + m * (255 - IR) * 0.3

m = mix, m принадлежит [0, 1].

Логика по каналам:

- красный выкручивается по ИК-яркости — растительность ярко-красная;
- зелёный приглушается остаточным синим — оригинальная зелень убирается;
- синему оставляется его вклад плюс инвертированный ИК — небо остаётся
  глубоким и не уходит в серое.

### Зерно

Аналогично Antique, но с другим коэффициентом:

    eta(x, y) ~ N(0, (grain * 25)^2)
    I'(x, y, c) = I_1(x, y, c) + eta(x, y) * mask(x, y)
    mask(x, y)  = clip( 1 - 4*(lum - 0.5)^2,  0.1,  1 )

## 6. Matte — овальная маска (smoothstep)

### Расстояние до центра

    a = (W / 2) * scale
    b = (H / 2) * scale
    d(x, y) = sqrt( ((x - cx) / a)^2 + ((y - cy) / b)^2 )

d < 1 — внутри эллипса, d = 1 — на границе, d > 1 — снаружи.

### Линейная маска

    t(x, y) = clip( (1 - d) / softness + 0.5,  0,  1 )

### Сглаживание smoothstep

    M(x, y) = t^2 * (3 - 2*t) = 3*t^2 - 2*t^3

Smoothstep даёт плавный переход с нулевой производной на концах —
мягче, чем линейная маска.

### Смешивание

    I'(x, y, c) = I(x, y, c) * M(x, y) + 255 * (1 - M(x, y))

## 7. Scratches — шум, пыль, царапины

### Зерно

    eta(x, y) ~ N(0, noise_sigma^2)
    I_1(x, y, c) = I(x, y, c) + eta(x, y)

### Пыль

С вероятностью dust на пиксель:

- с вероятностью 0.6*dust пиксель становится белым (245);
- с вероятностью 0.4*dust — чёрным (15);
- иначе — без изменений.

### Царапины

Каждая царапина с вероятностью vertical_bias — длинная вертикальная,
иначе — короткая произвольная. Параметры:

- вертикальная: x0 в [0, W), y0 в [0, H/3), L в [H/2, H), угол в [-0.1, 0.1];
- произвольная: x0 в [0, W), y0 в [0, H), L в [H/6, H/3), угол в [-0.5, 0.5].

Толщина t в {1, 2}. Затухание вдоль царапины:

    alpha(k) = linspace(1.0, 0.25, L),  k = 0..L-1

Каждый пиксель линии смешивается с белым:

    I'(y, x+k) = 255 * alpha(k) + I_1(y, x+k) * (1 - alpha(k))

## 8. Neon — неоновые контуры + свечение

### Яркость и лапласиан

    Y = 0.299 * R + 0.587 * G + 0.114 * B

Ядро свёртки:

    K = |  0   1   0 |
        |  1  -4   1 |
        |  0   1   0 |

    L(x, y) = сумма по dx,dy из {-1, 0, 1} от K(dy, dx) * Y(x + dx, y + dy)

Границы обрабатываются через np.pad(gray, 1, mode="edge").

### Контуры

    E(x, y) = clip( 3 * (|L(x, y)| - T), 0, 255 )

T = threshold — порог отсечения слабых контуров.

### Цветные контуры

    neon(x, y, c) = (E(x, y) / 255) * color(c)

### Свечение (box-blur)

Повторяем box-blur radius раз (каждый шаг — среднее по 5 точкам креста):

    blur(x, y, c) = ( up + down + left + right + center ) / 5

После radius итераций:

    glow(x, y, c) = blur^radius( neon )(x, y, c)

### Итог

    I'(x, y, c) = 0.2 * I(x, y, c) + neon(x, y, c)
                + glow_strength * glow(x, y, c)

Тёмный фон, цветные контуры с мягким свечением.

---

# Архитектура

filters/
├── base.py             ImageFilter (ABC) + get_filter
├── resize_bilinear.py  Resize
├── grayscale.py        RGB2GrayScale
├── antique_sepia.py    Antique
├── fade.py             FadeColor
├── film_infrared.py    InfraredFilm
├── matte_oval.py       Matte
├── scratches_dust.py   Scratches
└── neon_laplacian.py   Neon

ImageFilter — абстрактный класс с фабричным методом get_filter(name, **params).
Наследники реализуют apply_filter(image) -> image.

Фильтры используют только базовые матричные операции numpy. Высокоуровневые
функции OpenCV (cv2.resize, cv2.Laplacian, cv2.filter2D) не применяются.
I/O — через Pillow.