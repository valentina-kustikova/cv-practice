# Практическая работа №1. Обработка изображений

## Установка и запуск

    python -m venv .venv

    # Linux / macOS
    source .venv/bin/activate

    # Windows (cmd)
    .venv\Scripts\activate.bat

    pip install -r requirements.txt

    python main.py -i input.png -o output.png -f <filter> [параметры]

Скрипт выполняет: разбор аргументов командной строки -> чтение изображения
(с учётом EXIF-ориентации) -> применение фильтра -> сохранение результата.
Ошибки чтения/записи/неизвестного фильтра логируются и возвращают код 1.

## Текстуры

Фильтр `scratches` умеет накладывать готовые текстуры из папки
`textures/` (рядом с `main.py`):

    textures/
    ├── scratches.png   # царапины
    ├── dust.png        # пыль
    └── grain.png       # зерно / старая бумага

Если файла нет — фильтр пишет `logging.warning` и пропускает этот слой
(graceful fallback). Каждая текстура читается через `read_image`
(конвертация в RGB + EXIF-поворот), затем переводится в оттенки серого
по BT.601 и ресайзится под размер изображения **методом ближайшего
соседа** (nearest neighbor).

## Фильтры

| Фильтр    | Параметры                                                                 | Описание                              |
|-----------|---------------------------------------------------------------------------|---------------------------------------|
| resize    | --width, --height                                                         | билинейное изменение разрешения       |
| grayscale | —                                                                         | оттенки серого (BT.601)               |
| antique   | --strength, --grain, --seed                                               | сепия + виньетка + зерно              |
| fade      | --strength, --black-lift                                                  | выцветание, сжатие диапазона          |
| film      | --mix, --grain, --seed, --veil, --black-lift, --tint                      | ИК-монохром + вуаль + зерно           |
| matte     | --softness, --scale, --mask-width, --mask-height, --center-x, --center-y  | белая овальная рамка (smoothstep)     |
| scratches | --n-scratches, --noise-sigma, --seed, --dust, --vertical-bias,            | текстуры + царапины + пыль            |
|           | --texture-strength                                                        |                                       |
| neon      | --threshold, --glow-radius, --glow-strength, --halo-radius,               | неоновые контуры + свечение + гало    |
|           | --halo-strength                                                           |                                       |

## Примеры

    # resize
    python main.py -i input.png -o out/resize_800x600.png -f resize --width 800 --height 600
    python main.py -i input.png -o out/resize_fullhd.png -f resize --width 1920 --height 1080
    python main.py -i input.png -o out/resize_thumb.png -f resize --width 512 --height 512

    # grayscale
    python main.py -i input.png -o out/grayscale.png -f grayscale

    # antique
    python main.py -i input.png -o out/antique.png -f antique
    python main.py -i input.png -o out/antique_strong.png -f antique --strength 1.5 --grain 0.4
    python main.py -i input.png -o out/antique_light.png -f antique --strength 0.4 --grain 0.05
    python main.py -i input.png -o out/antique_no_grain.png -f antique --strength 1.0 --grain 0.0
    python main.py -i input.png -o out/antique_seed7.png -f antique --strength 1.0 --grain 0.3 --seed 7

    # fade
    python main.py -i input.png -o out/fade.png -f fade
    python main.py -i input.png -o out/fade_strong.png -f fade --strength 0.9 --black-lift 60
    python main.py -i input.png -o out/fade_light.png -f fade --strength 0.3 --black-lift 15
    python main.py -i input.png -o out/fade_lift.png -f fade --strength 0.0 --black-lift 50
    python main.py -i input.png -o out/fade_desat.png -f fade --strength 1.0 --black-lift 0

    # film
    python main.py -i input.png -o out/film.png -f film
    python main.py -i input.png -o out/film_cyan.png -f film --mix 1.0 --veil 0.3 --tint cyan
    python main.py -i input.png -o out/film_magenta.png -f film --mix 0.8 --veil 0.25 --tint magenta
    python main.py -i input.png -o out/film_sepia.png -f film --mix 0.7 --veil 0.3 --tint sepia
    python main.py -i input.png -o out/film_weak.png -f film --mix 0.2 --veil 0.05 --grain 0.1
    python main.py -i input.png -o out/film_max.png -f film --mix 1.0 --veil 0.35 --black-lift 10 --grain 0.5 --tint cyan

    # matte
    python main.py -i input.png -o out/matte.png -f matte
    python main.py -i input.png -o out/matte_soft.png -f matte --softness 0.4 --scale 0.9
    python main.py -i input.png -o out/matte_hard.png -f matte --softness 0.05 --scale 0.8
    python main.py -i input.png -o out/matte_small.png -f matte --scale 0.6 --softness 0.2
    python main.py -i input.png -o out/matte_big.png -f matte --scale 1.1 --softness 0.15
    python main.py -i input.png -o out/matte_custom.png -f matte --mask-width 400 --mask-height 300 --center-x 500 --center-y 350 --softness 0.2
    python main.py -i input.png -o out/matte_portrait.png -f matte --mask-width 300 --mask-height 500 --softness 0.25

    # scratches
    python main.py -i input.png -o out/scratches.png -f scratches
    python main.py -i input.png -o out/scratches_many.png -f scratches --n-scratches 30 --noise-sigma 10 --dust 0.006 --vertical-bias 0.7
    python main.py -i input.png -o out/scratches_light.png -f scratches --n-scratches 4 --noise-sigma 6 --dust 0.002 --texture-strength 0.2
    python main.py -i input.png -o out/scratches_only.png -f scratches --n-scratches 20 --noise-sigma 0 --dust 0 --texture-strength 0
    python main.py -i input.png -o out/scratches_dust.png -f scratches --n-scratches 0 --noise-sigma 20 --dust 0.01 --texture-strength 0
    python main.py -i input.png -o out/scratches_tex.png -f scratches --n-scratches 0 --noise-sigma 0 --dust 0 --texture-strength 0.6
    python main.py -i input.png -o out/scratches_vert.png -f scratches --n-scratches 15 --vertical-bias 0.95 --texture-strength 0.3
    python main.py -i input.png -o out/scratches_chaos.png -f scratches --n-scratches 25 --vertical-bias 0.0 --texture-strength 0.4
    python main.py -i input.png -o out/scratches_max.png -f scratches --n-scratches 40 --noise-sigma 25 --dust 0.02 --vertical-bias 0.5 --texture-strength 0.7

    # neon
    python main.py -i input.png -o out/neon.png -f neon
    python main.py -i input.png -o out/neon_thin.png -f neon --threshold 60 --glow-radius 1 --glow-strength 1.5 --halo-radius 4 --halo-strength 0.6
    python main.py -i input.png -o out/neon_thick.png -f neon --threshold 15 --glow-radius 3 --glow-strength 1.0 --halo-radius 8 --halo-strength 1.0
    python main.py -i input.png -o out/neon_halo.png -f neon --threshold 25 --glow-radius 2 --glow-strength 1.2 --halo-radius 12 --halo-strength 1.5
    python main.py -i input.png -o out/neon_nohalo.png -f neon --threshold 30 --halo-radius 0 --glow-strength 1.5
    python main.py -i input.png -o out/neon_max.png -f neon --threshold 20 --glow-radius 3 --glow-strength 2.0 --halo-radius 10 --halo-strength 1.5
    python main.py -i input.png -o out/neon_min.png -f neon --threshold 80 --glow-radius 1 --glow-strength 0.8 --halo-radius 3 --halo-strength 0.3

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

## 5. Film — инфракрасная плёнка + вуаль + зерно

Эффект имитирует съёмку в ближнем ИК: живая зелень (хлорофилл) начинает
отражать очень сильно и «выгорает» в белое, небо и синие объекты —
уходят в тёмное, кадр становится почти монохромным с эмульсионным
свечением и цветовой вуалью.

### ИК-монохром

Изолируем зелёный вклад, подавляя красный и синий:

    green_mask(x, y) = clip( 2*G - 0.5*R - 0.5*B,  0,  255 )

Базовая ИК-яркость (зелёный доминирует, синий уничтожается):

    ir_base(x, y) = 0.8*G + 0.4*R - 0.6*B
    ir_gray(x, y) = clip( ir_base + 1.2 * green_mask,  0,  255 )

### S-кривая контраста

Нормализуем и применяем сигмоиду:

    u(x, y)     = ir_gray(x, y) / 255
    s(u)        = 1 / (1 + exp( -10 * (u - 0.45) ))
    ir_gray'(x, y) = s(u) * 255

Света травы разгоняются до 255, тени становятся угольными.

### Смешивание с обычным ЧБ

    standard_gray(x, y) = 0.299*R + 0.587*G + 0.114*B
    base_mono(x, y)     = (1 - m) * standard_gray + m * ir_gray'

m = mix, m принадлежит [0, 1].

    out(x, y, c) = base_mono(x, y)   для всех c

### Подъём чёрного

    f_min = lift
    out_1 = f_min + out * (255 - f_min) / 255

### Эмульсионное свечение (halo)

Box-blur по кресту (5 точек):

    blur(x, y, c) = ( up + down + left + right + center ) / 5

    out_2 = 0.4 * out_1 + 0.6 * blur(out_1)

Баланс смещён в сторону свечения — вокруг яркой травы появляется
мягкий ореол, характерный для ИК-плёнки.

### Цветовая вуаль

    veil_color = (130, 160, 170)   # cyan
               | (170, 140, 160)   # magenta
               | (160, 145, 125)   # sepia
    out_3 = (1 - v) * out_2 + v * veil_color

v = veil, v принадлежит [0, 1].

### Зерно

    lum(x, y)  = base_mono(x, y) / 255
    mask       = clip( 1 - 4*(lum - 0.5)^2,  0.1,  1 )
    eta(x, y) ~ N(0, (grain * 30)^2)
    out'(x, y, c) = out_3(x, y, c) + eta(x, y) * mask(x, y)

## 6. Matte — овальная маска (smoothstep)

### Расстояние до центра

    a = mask_width   (или W/2 * scale, если не задано)
    b = mask_height  (или H/2 * scale, если не задано)
    cx = center_x    (или (W-1)/2, если не задано)
    cy = center_y    (или (H-1)/2, если не задано)

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

## 7. Scratches — текстуры, шум, пыль, царапины

### Зерно

    eta(x, y) ~ N(0, noise_sigma^2)
    I_1(x, y, c) = I(x, y, c) + eta(x, y)

### Наложение готовых текстур

Если в папке `textures/` есть файлы, они читаются через `read_image`
(EXIF + RGB), переводятся в оттенки серого по BT.601 и ресайзятся
под изображение **nearest neighbor**:

    gray_tex(x, y) = 0.299*R + 0.587*G + 0.114*B   (из файла)
    tex = gray_tex / 255

    # scratches.png — multiply (тёмные участки затемняют)
    I_2 = I_1 * ( (1 - a) + a * tex )
    a = texture_strength

    # dust.png — screen (светлые пятна подсвечивают)
    I_3 = I_2 + (1 - I_2 / 255) * dust_tex * (a * 0.5) * 255

    # grain.png — overlay (сдвиг вокруг 128)
    I_4 = I_3 + (grain_tex - 0.5) * (a * 0.3) * 40

Если файла нет — слой пропускается, в лог пишется `logging.warning`.

### Мягкая пыль

Число пятен:

    n_spots = int( dust * H * W * 0.02 )

Каждое пятно — квадрат радиуса r ∈ {1, 2}:

    val = uniform(200, 245)   # белое, с вероятностью 0.5
        | uniform(10, 50)     # чёрное, иначе
    I_5(patch) = 0.6 * I_4(patch) + 0.4 * val

### Хаотичные царапины

Каждая царапина — ломаная с блужданием угла:

    x, y      = random start
    length    = randint( max(20, H/6), max(40, H) )
    ang       = (вертикальный: pi/2 + U(-0.25, 0.25))
              | (произвольный: U(0, 2*pi))
    thickness = 1 (85%) | 2 (15%)
    base_alpha = U(0.08, 0.30)
    color      = U(190, 235)

На каждом шаге t = 0..length-1:

    ang += N(0, 0.06)          # блуждание
    x += cos(ang); y += sin(ang)
    fade = 1 - t / length
    a    = base_alpha * fade * U(0.6, 1.0)
    I_6(y, x) = color * a + I_5(y, x) * (1 - a)

Если пиксель выходит за границы — царапина обрывается.
Никаких ярко-белых толстых полос: цвет 190–235, alpha <= 0.30.

## 8. Neon — неоновые контуры + свечение + гало

### Яркость и лапласиан

    Y = 0.299 * R + 0.587 * G + 0.114 * B

Ядро свёртки (8-связный лапласиан):

    K = |  1   1   1 |
        |  1  -8   1 |
        |  1   1   1 |

    L(x, y) = сумма по dx,dy из {-1, 0, 1} от K(dy, dx) * Y(x + dx, y + dy)

Границы обрабатываются через np.pad(gray, 1, mode="edge").

### Маска контура

    edge_mask(x, y) = clip( (|L(x, y)| - T) * 3,  0,  255 ) / 255

T = threshold — порог отсечения слабых контуров.
edge_mask принадлежит [0, 1].

Цвет неона нормируется:

    c = color / 255

### Слои неонового света

**1) Жёсткое ядро контура:**

    neon_edge(x, y) = edge_mask(x, y) * c

**2) Внутреннее мягкое свечение** — скользящее среднее через cumsum:

    box_blur(img, r):
        padded_x = pad(img, ((0,0), (r,r)), mode='edge')
        cum_x    = cumsum(padded_x, axis=1)
        blur_x   = (cum_x[:, 2r:] - cum_x[:, :-2r]) / (2r)
        padded_y = pad(blur_x, ((r,r), (0,0)), mode='edge')
        cum_y    = cumsum(padded_y, axis=0)
        blur_y   = (cum_y[2r:, :] - cum_y[:-2r, :]) / (2r)
        return blur_y

    inner(x, y) = box_blur( edge_mask,  glow_radius * 2 )
    neon_inner(x, y) = inner(x, y) * c * glow_strength

**3) Внешнее гало** — широкое размытие маски контура:

    r_wide = halo_radius * 4
    wide(x, y) = box_blur( edge_mask,  r_wide )
    halo_mask(x, y) = clip( wide(x, y) - inner(x, y) * 0.2,  0,  1 )
    neon_halo(x, y) = halo_mask(x, y) * c * halo_strength

Вычитание `inner * 0.2` аккуратно вырезает центр — остаётся мягкое
внешнее облако вокруг контура.

### Блендинг Screen

Слои складываются в режиме screen (без пересветов):

    neon_total = 1 - (1 - neon_edge) * (1 - neon_inner) * (1 - neon_halo)

Фон притемняется до 15%:

    bg = (I / 255) * 0.15

Итог — тоже screen:

    I' = (1 - (1 - bg) * (1 - neon_total)) * 255

Тёмный фон, яркое ядро, мягкое свечение и объёмное гало вокруг.

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

    textures/
    ├── scratches.png       (опционально)
    ├── dust.png            (опционально)
    └── grain.png           (опционально)

ImageFilter — абстрактный класс с фабричным методом get_filter(name, **params).
Наследники реализуют apply_filter(image) -> image.

Фильтры используют только базовые матричные операции numpy. Высокоуровневые
функции OpenCV (cv2.resize, cv2.Laplacian, cv2.filter2D) не применяются.
I/O — через Pillow (с учётом EXIF-ориентации).