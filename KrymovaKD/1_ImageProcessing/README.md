# Практическая работа № 1. Обработка изображений

## 1. Цель работы

Цель работы — разработать библиотеку фильтров для обработки изображений на Python с использованием OpenCV и NumPy.

В рамках работы реализованы восемь фильтров: изменение размера изображения, преобразование в оттенки серого, эффект сепии, выцветание, плёночный эффект, белая матовая рамка, состаривание фотографии и неоновое свечение контуров.

Фильтры применяются к изображениям через интерфейс командной строки.

## 2. Используемые технологии

- **Python** — язык программирования.
- **NumPy** — работа с массивами пикселей и выполнение математических операций.
- **OpenCV** — чтение и сохранение изображений.
- **argparse** — обработка аргументов командной строки.

При реализации фильтров используются базовые операции над массивами. Это позволяет самостоятельно реализовать алгоритмы обработки изображений без применения готовых высокоуровневых функций фильтрации.

## 3. Структура проекта

```text
cv-practice/
├── .venv/
└── KrymovaKD/
    └── 1_ImageProcessing/
        ├── main.py
        ├── cli.py
        ├── image_io.py
        ├── requirements.txt
        ├── README.md
        ├── filters/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── geometry.py
        │   ├── color.py
        │   └── effects.py
        ├── images/
        │   └── test.jpg
        └── output/
```

Назначение основных файлов:

- `main.py` — последовательность выполнения программы: получение аргументов, чтение изображения, применение фильтра и сохранение результата.
- `cli.py` — настройка аргументов командной строки.
- `image_io.py` — чтение и сохранение изображений.
- `filters/base.py` — базовый класс фильтра и механизм выбора реализации.
- `filters/geometry.py` — геометрические преобразования.
- `filters/color.py` — цветовые преобразования.
- `filters/effects.py` — визуальные эффекты.
- `images/` — входные изображения.
- `output/` — результаты обработки.

## 4. Установка и запуск

### 4.1. Подготовка окружения

Перейдите в корневую папку репозитория и активируйте виртуальное окружение:

```bash
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install -r KrymovaKD/1_ImageProcessing/requirements.txt
```

### 4.2. Общий формат команды

Команды выполняются из корневой папки репозитория:

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/result.jpg \
  --filter grayscale
```

Параметры:

- `--input` — путь к входному изображению.
- `--output` — путь сохранения результата.
- `--filter` — название применяемого фильтра.

Для просмотра доступных аргументов выполни:

```bash
python KrymovaKD/1_ImageProcessing/main.py --help
```

### 4.3. Примеры запуска фильтров

**Изменение размера:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/resized.jpg \
  --filter resize
```

**Оттенки серого:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/gray.jpg \
  --filter grayscale
```

**Сепия:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/antique.jpg \
  --filter antique
```

**Выцветание:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/fade.jpg \
  --filter fade
```

**Плёночный эффект:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/film.jpg \
  --filter film
```

**Белая матовая рамка:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/matte.jpg \
  --filter matte
```

**Состаренная фотография:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/noise.jpg \
  --filter noise
```

**Неоновый эффект:**

```bash
python KrymovaKD/1_ImageProcessing/main.py \
  --input KrymovaKD/1_ImageProcessing/images/test.jpg \
  --output KrymovaKD/1_ImageProcessing/output/neon.jpg \
  --filter neon
```

Дополнительные параметры фильтров и их значения по умолчанию можно посмотреть с помощью `--help` и в файле `cli.py`.

## 5. Описание алгоритмов

В обозначениях ниже `I` — исходное изображение, `I'` — результат обработки, а `B`, `G`, `R` — синий, зелёный и красный цветовые
