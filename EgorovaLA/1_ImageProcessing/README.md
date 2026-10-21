# Лабораторная работа №1. Обработка изображений, фильтры

Библиотека фильтров изображений на Python.

## Структура проекта

```
lab1/
├── main.py            # скрипт запуска
├── filters/
│   ├── __init__.py    # подключение всех фильтров
│   ├── base.py        # базовый класс ImageFilter и фабрика get_filter()
│   ├── utils.py       # общие функции: to_uint8, to_gray, blur, screen
│   ├── resize.py      # Resize       — изменение разрешения
│   ├── grayscale.py   # GrayScale    — оттенки серого
│   ├── antique.py     # Antique      — сепия
│   ├── fade.py        # FadeColor    — выцветание
│   ├── infrared.py    # InfraredFilm — инфракрасная плёнка
│   ├── matte.py       # Matte        — овальная рамка
│   ├── oldphoto.py    # OldPhoto     — состаренное фото
│   └── neon.py        # Neon         — неоновые контуры
├── requirements.txt
├── images/            # тестовые изображения
└── results/           # результаты
```

## Установка

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Запуск

```bash
python main.py -i <изображение> -f <фильтр> [<фильтр> ...] [-o <файл или папка>] [-p ключ=значение ...]
```

| Аргумент | Описание |
|---|---|
| `-i`, `--input` | входное изображение |
| `-f`, `--filter` | один или несколько фильтров через пробел, либо `all` — все фильтры |
| `-o`, `--output` | файл результата (для одного фильтра) или папка. По умолчанию `results` |
| `-p`, `--param` | параметр фильтра `ключ=значение`, можно указывать несколько раз. Только при одном фильтре |

Если `-o` — папка, результаты сохраняются как `<имя>_<фильтр>.<расширение>`.

**Один фильтр** (с параметрами по умолчанию или со своими):

```bash
python main.py -i images/cat.png -f neon
python main.py -i images/cat.png -f neon -p color=cyan -p glow=1.5
python main.py -i images/cat.png -f neon -o results/cat_neon.jpg
```

**Несколько фильтров или все сразу** (параметры по умолчанию, каждый
фильтр применяется к исходному изображению отдельно):

```bash
python main.py -i images/cat.png -f antique fade matte
python main.py -i images/cat.png -f all
```

### Порядок работы `main.py`

1. `cli_argument_parser()` — разбор аргументов (`argparse`).
2. `read_image()` — чтение изображения в матрицу `H×W×3` (BGR, `uint8`).
   Используется `np.fromfile` + `cv2.imdecode`, чтобы работали пути с кириллицей.
3. `ImageFilter.get_filter(name, **params)` — создание фильтров по имени.
4. `apply_filter(image)` — применение фильтра, `save_image()` — сохранение.

| Ошибка | Сообщение | Код возврата |
|---|---|---|
| файл не найден или не является изображением | `Ошибка чтения: ...` | 1 |
| неизвестный фильтр, неверный параметр, `-p` при нескольких фильтрах | `Ошибка фильтра: ...` | 2 |
| не удалось сохранить файл | `Ошибка сохранения: ...` | 3 |

### Устройство фильтров

`ImageFilter` — абстрактный класс с методом `apply_filter(image)` и
статическим фабричным методом `get_filter(name, **params)`. Каждый фильтр —
наследник с атрибутом `name`. Фабрика находит нужный класс среди наследников
(`ImageFilter.__subclasses__()`) и создаёт его с переданными параметрами.
Чтобы добавить фильтр, достаточно создать новый класс-наследник и импортировать
его в `filters/__init__.py`.

---

## Математическое описание фильтров

$I(x, y)$ — интенсивность пикселя, $R, G, B$ — каналы в $[0, 255]$,
$W \times H$ — размер изображения. После вычислений значения обрезаются
до $[0, 255]$ и округляются.

### Вспомогательные функции (`utils.py`)

**Яркость** (ITU-R BT.601), `to_gray`:

$$Y = 0.299R + 0.587G + 0.114B.$$

**Размытие**, `blur`. Усреднение в окне $(2r+1)$ по строкам и столбцам через
кумулятивные суммы $S(n) = \sum_{k \le n} I(k)$:

$$\bar I(n) = \frac{S(n + r) - S(n - r - 1)}{2r + 1}.$$

Три таких прохода подряд по центральной предельной теореме дают приближение
гауссова размытия с $\sigma \approx r$.

**Наложение «Экран»**, `screen`:

$$\mathrm{screen}(A, B) = 255 - \frac{(255 - A)(255 - B)}{255}.$$

### 1. Изменение разрешения — `resize`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `width`, `height` | — | новый размер; если задан один, второй считается по пропорциям |
| `scale` | `0.5` | коэффициент масштабирования, если размер не задан |

Для пикселя $(x', y')$ нового изображения точка в исходном:

$$x = (x' + 0.5)\frac{W}{W'} - 0.5, \qquad y = (y' + 0.5)\frac{H}{H'} - 0.5.$$

Билинейная интерполяция по четырём соседям, $x_0 = \lfloor x \rfloor$,
$y_0 = \lfloor y \rfloor$, $d_x = x - x_0$, $d_y = y - y_0$:

$$I' = (1-d_y)\big[(1-d_x)I_{00} + d_x I_{01}\big] + d_y\big[(1-d_x)I_{10} + d_x I_{11}\big].$$

```bash
python main.py -i images/cat.png -f resize -p width=200
```

### 2. Оттенки серого — `grayscale`

$I' = Y = 0.299R + 0.587G + 0.114B$. Параметров нет.

```bash
python main.py -i images/cat.png -f grayscale
```

### 3. Антиквариат (сепия) — `antique`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `intensity` | `1.0` | сила эффекта, $[0, 1]$ |

$$
\begin{pmatrix} R_s \\ G_s \\ B_s \end{pmatrix} =
\begin{pmatrix}
0.393 & 0.769 & 0.189 \\
0.349 & 0.686 & 0.168 \\
0.272 & 0.534 & 0.131
\end{pmatrix}
\begin{pmatrix} R \\ G \\ B \end{pmatrix},
\qquad I' = (1-k)\,I + k\,I_s.
$$

Суммы строк матрицы убывают от R к B (1.35, 1.20, 0.94), поэтому изображение
уходит в тёплые коричневые тона. В коде матрица записана для порядка каналов BGR.

```bash
python main.py -i images/cat.png -f antique -p intensity=0.8
```

### 4. Выцветание — `fade`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `saturation` | `0.5` | доля сохраняемой насыщенности, $[0, 1]$ |
| `fade` | `0.4` | сила выцветания, $[0, 1]$ |

Снижение насыщенности, затем сжатие диапазона яркостей (чёрный становится
серым, белый тускнеет):

$$C_1 = Y + s\,(C - Y), \qquad C_2 = L + C_1\frac{H_i - L}{255}, \qquad L = 90f,\ H_i = 255 - 35f.$$

```bash
python main.py -i images/cat.png -f fade -p saturation=0.3 -p fade=0.6
```

### 5. Инфракрасная плёнка — `infrared`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `mode` | `color` | `color` — ложные цвета, `bw` — чёрно-белая ИК-плёнка |
| `glow` | `0.4` | сила ореола вокруг светлых участков |

ИК-яркость оценивается по видимым каналам. Растительность отражает ИК очень
сильно (признак — избыток зелёного), кожа и красные объекты — примерно как
красный свет, небо и вода поглощают ИК (признак — избыток синего):

$$IR = 0.9R + 0.2G - 0.2B + 1.2\max(2G - R - B,\ 0) - 0.7\max\!\Big(B - \frac{R + G}{2},\ 0\Big).$$

- `color` — ложные цвета, как у плёнки Kodak Aerochrome: $R' = IR,\ G' = R,\ B' = G$.
  Зелень становится красной, красное — жёлтым, небо — тёмно-синим.
- `bw` — $R' = G' = B' = IR$: белая листва, тёмное небо.

Ореол: яркие участки $S = (I - 160)\cdot\frac{255}{95}$ размываются с
$\sigma = \max(W, H)/80$ и накладываются: $I' = \mathrm{screen}(I,\ \text{glow}\cdot\mathrm{blur}(S))$.

```bash
python main.py -i images/aloe.jpg -f infrared
python main.py -i images/aloe.jpg -f infrared -p mode=bw
```

### 6. Маска Matte — `matte`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `size` | `1.05` | размер овала относительно кадра |
| `softness` | `0.3` | ширина плавного перехода, $[0, 1]$ |
| `color` | `255` | яркость рамки (255 — белая, 0 — чёрная) |

$$d = \sqrt{\left(\frac{x - W/2}{a}\right)^2 + \left(\frac{y - H/2}{b}\right)^2},
\qquad a = \text{size}\cdot\frac{W}{2},\ b = \text{size}\cdot\frac{H}{2}.$$

Внутри овала $d < 1$. Маска плавно растёт у границы (smoothstep):

$$t = \mathrm{clip}\!\left(\frac{d - 1 + \text{softness}}{\text{softness}}\right),
\qquad \alpha = 3t^2 - 2t^3, \qquad I' = (1 - \alpha)\,I + \alpha\cdot\text{color}.$$

```bash
python main.py -i images/cat.png -f matte -p size=0.9 -p color=0
```

### 7. oldphoto — состаренное фото
Параметры: `noise=0.4`, `scratches=20`, `spots=15`, `creases=1`,
`vignette=0.6` ($v$), `seed` (чтобы результат повторялся).
 
1. Размытие, сепия и выцветание: $I = 25 + 0.8\,I_{sepia}$.
2. Виньетка: $I \leftarrow I\,(1 - 0.6\,v\,r^2)$, $r$ — расстояние до центра (1 у края).
3. Зерно: $I \leftarrow I + \mathcal N(0,\ (45\cdot\text{noise})^2)$.
4. Заломы: для прямой с нормалью $n$ расстояние $d = (p - p_0)\cdot n$;
   светлая полоса $(235 - I)\cdot 0.55\,e^{-(d/w)^2}$ и тень рядом.
5. Царапины: вертикальные линии со случайным сдвигом $x_{k+1} = x_k + \varepsilon$,
   $\varepsilon \in \{-1, 0, 1\}$.
6. Пятна: круги с мягким краем $\alpha = \mathrm{clip}(1.5 - \rho/r)$.
7. Потёртый край: пиксели у границы кадра заменяются цветом бумаги.

```bash
python main.py -i images/cat.png -f oldphoto -p seed=7
python main.py -i images/cat.png -f oldphoto -p noise=0.7 -p scratches=50 -p creases=3 -p vignette=1
```

### 8. Неоновый эффект — `neon`

Параметры: `color=rainbow` (`source`, `cyan`, `pink`, `green`, `yellow`),
`threshold=0.12` ($t$), `glow=1.2`, `background=0.15`.
 
Контуры — градиент яркости оператором Собеля:
 
$$K_x = \begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix},\ K_y = K_x^T, \qquad
|G| = \sqrt{G_x^2 + G_y^2}, \qquad E = \mathrm{clip}\Big(\frac{|G| - t}{1 - t}\Big).$$
 
Цвет линий $T$: в режиме `rainbow` зависит от направления контура
$\theta = \mathrm{atan2}(G_y, G_x)$, в `source` — исходный цвет пикселя.
Свечение — сумма размытых копий линий:
$N = L + \text{glow}\cdot(0.9\,\mathrm{blur}_2 + 0.7\,\mathrm{blur}_6 + 0.5\,\mathrm{blur}_{14})(L)$,
$L = T\cdot E$. Итог: $I' = \text{background}\cdot I \oplus N$.

```bash
python main.py -i images/cat.png -f neon
python main.py -i images/aloe.jpg -f neon -p color=cyan -p glow=1.5
```