# Практическая работа №1 — обработка изображений с OpenCV

## 1. Цель

Разработать библиотеку фильтров на Python с использованием базовых матричных операций над изображением в OpenCV/NumPy.

Реализованы 8 фильтров без использования высокоуровневых функций OpenCV. Есть CLI с выбором фильтра и его параметров.

## 2. Структура

```text
KulikAI/
└── 1_ImageProcessing/
    ├── main.py
    ├── image_filters.py
    ├── README.md
    └── requirements.txt
```

`ImageFilter` — абстрактный класс. Конкретные фильтры реализованы отдельными наследниками.

## 3. Установка в Anaconda Prompt

Перейдите в каталог лабораторной:

```bat
cd ...\KulikAI\1_ImageProcessing
```

Создайте окружение:

```bat
conda create -n cv_lab1 python=3.11 -y
conda activate cv_lab1
pip install -r requirements.txt
```

## 4. Общий формат запуска

```bat
python main.py --input INPUT --output OUTPUT --filter FILTER [ПАРАМЕТРЫ]
```

Доступные фильтры:

```text
resize
grayscale
antique
fade
film
matte
noise
neon
all
```

Полную справку можно посмотреть:

```bat
python main.py --help
```

### Один фильтр

```bat
python main.py -i photo.jpg -o result.png -f grayscale
```

### Resize

```bat
python main.py -i photo.jpg -o result.png -f resize --width 1200 --height 800 --interpolation bilinear
```

Доступны `nearest` и `bilinear`. 

### Antique

```bat
python main.py -i photo.jpg -o result.png -f antique --antique-strength 0.8
```

### Fade Color

```bat
python main.py -i photo.jpg -o result.png -f fade --fade-alpha 0.35 --fade-color 235,215,175
```

Цвет задается в формате `R,G,B`.

### Film

```bat
python main.py -i photo.jpg -o result.png -f film --film-gamma 0.85 --film-grain 0.06
```

### Matte

```bat
python main.py -i photo.jpg -o result.png -f matte --matte-radius 0.58 --matte-strength 1.0 --matte-softness 2.2
```

### Царапины и шум

```bat
python main.py -i photo.jpg -o result.png -f noise --noise-amount 0.10 --scratch-density 0.03
```

### Neon

```bat
python main.py -i photo.jpg -o result.png -f neon --neon-strength 1.5 --neon-blur 0.8
```

## 5. Математическое описание алгоритмов

Изображение рассматривается как матрица:

\[
I \in [0,255]^{H\times W\times 3},
\]

где третий индекс соответствует каналам B, G, R.

### 6.1. Изменение разрешения

реализована bilinear interpolation.

Для каждой координаты результата вычисляются непрерывные координаты исходного изображения:

\[
x = j\frac{W_s-1}{W_d-1}, \qquad
y = i\frac{H_s-1}{H_d-1}.
\]

Пусть:

\[
x_0=\lfloor x\rfloor,\; x_1=\min(x_0+1,W_s-1),
\]

\[
y_0=\lfloor y\rfloor,\; y_1=\min(y_0+1,H_s-1).
\]

Тогда:

\[
I'(i,j)=
(1-t_x)(1-t_y)I(y_0,x_0)
+t_x(1-t_y)I(y_0,x_1)
+(1-t_x)t_yI(y_1,x_0)
+t_xt_yI(y_1,x_1).
\]

Все координаты и веса строятся как матрицы NumPy; поэлементного Python-цикла нет.

### 6.2. Grayscale

Для изображения BGR используется линейная комбинация каналов:

\[
Y = 0.114B + 0.587G + 0.299R.
\]

Получается матрица интенсивностей \(H\times W\).

### 6.3. Antique

Используется сепия-преобразование:

\[
R' = 0.393R+0.769G+0.189B,
\]

\[
G' = 0.349R+0.686G+0.168B,
\]

\[
B' = 0.272R+0.534G+0.131B.
\]

Для параметра силы \(\alpha\):

\[
I'=(1-\alpha)I+\alpha I_{sepia},
\qquad 0\le \alpha\le1.
\]

### 6.4. Fade Color

Пусть \(C\) — выбранный RGB-цвет выцветания, а \(\alpha\) — сила эффекта. Тогда:

\[
I'=(1-\alpha)I+\alpha C.
\]

Это линейная интерполяция между исходным изображением и заданным цветом.

### 6.5. Film

Сначала выполняется линейное смешивание каналов:

\[
B_f=0.95B+0.05G,
\]

\[
G_f=0.12B+0.78G+0.10R,
\]

\[
R_f=0.08G+0.92R.
\]

Затем гамма-преобразование:

\[
I_f'=I_f^\gamma
\]

после нормировки интенсивностей в диапазон \([0,1]\).

Для имитации зерна добавляется случайная матрица гауссовского шума:

\[
I_f'' = I_f' + N(0,\sigma^2).
\]

### 6.6. Matte

Центральная область сохраняется, а края изображения постепенно смешиваются с белым цветом.

Введем нормированные координаты:

\[
x,y\in[-1,1]
\]

и эллиптический радиус:

\[
r=\sqrt{x^2+y^2}.
\]

Для порога \(r_0\):

\[
t=\operatorname{clip}
\left(
\frac{r-r_0}{1-r_0},0,1
\right).
\]

С учетом параметра мягкости \(p\):

\[
m=t^p.
\]

Итог:

\[
I'=(1-m)I+m\cdot255.
\]

Таким образом, по краям получается белая овальная рамка.

### 6.7. Царапины и шум

Основной шум задается матрицей:

\[
N_{ij}\sim N(0,\sigma^2).
\]

К ней добавляется векторизованное поле полос, построенное по синусоидальным функциям координаты \(x\). Пороговая функция формирует длинные вертикальные области, визуально похожие на царапины.

### 6.8. Neon

Изображение переводится в яркость:

\[
Y=0.114B+0.587G+0.299R.
\]

Далее к матрице яркости применяются операторы Собеля:

\[
G_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\qquad
G_y=G_x^T.
\]

Свертка с этими матрицами дает:

\[
M=\sqrt{G_x^2+G_y^2}.
\]

После нормировки \(M\) используется как маска свечения. Фон затемняется, а контуры окрашиваются в неоновый цвет.

**

## 8. Последовательность выполнения программы

1. разбор аргументов командной строки;
2. чтение входного изображения и обработка исключений;
3. применение выбранного фильтра;
4. сохранение результата.

Это реализовано в `main.py`.
