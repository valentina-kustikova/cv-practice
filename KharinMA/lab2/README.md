# Практическая работа №2. Детектирование объектов на изображениях с использованием библиотеки OpenCV

## Описание проекта

Данный проект реализует систему автоматического детектирования транспортных средств на последовательности кадров видео. Система построена на базе модуля DNN библиотеки OpenCV и поддерживает работу с тремя различными архитектурами нейронных сетей.

---

## Структура проекта

```
lab2/
├── main.py                    # Главное приложение
├── base_detector.py           # Абстрактный базовый класс детектора
├── yolo_v3.py                 # Реализация YOLOv3-tiny
├── yolo_v4.py                 # Реализация YOLOv4-tiny
├── ssd_mobilenet_v2.py        # Реализация SSD MobileNet V2
├── data_reader.py             # Загрузчик данных и аннотаций
├── metrics.py                 # Вычисление метрик качества
├── README.md
├── data/                      # Директория с данными
│   ├── imgs_MOV03478/         # Кадры видео
│   └── mov03478.txt           # Файл аннотаций
└── models/                    # Директория с весами моделей
    ├── yolov3/
    ├── yolov4/
    └── mobilenet/
```

---

## Установка и настройка

### Зависимости

```bash
pip install opencv-python numpy
```

### Подготовка данных

В директорию `data/` необходимо поместить:

1. **Кадры видео** — папка `imgs_MOV03478/` с изображениями в формате `.jpg` или `.png`
2. **Файл аннотаций** — `mov03478.txt` с разметкой в формате:
   ```
   frame_id;x_min;y_min;x_max;y_max;class_name
   ```

### Скачивание моделей

#### YOLOv3-tiny

```bash
mkdir -p models/yolov3
cd models/yolov3

# Конфигурация сети
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg

# Веса (предобученные на COCO)
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# Список классов
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

#### YOLOv4-tiny

```bash
mkdir -p models/yolov4
cd models/yolov4

# Конфигурация сети
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# Веса (предобученные на COCO)
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

# Список классов
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

#### SSD MobileNet V2

```bash
mkdir -p models/mobilenet
cd models/mobilenet

# Скачать архив с моделью
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

# Копировать frozen graph
cp ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb .

# Скачать pbtxt для OpenCV DNN
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt -O ssd_mobilenet_v2.pbtxt
```

---

## Архитектура детекторов

Все детекторы наследуются от абстрактного класса `BaseDetector`, который определяет единый интерфейс:

- `_preprocess(image)` — подготовка изображения для нейросети
- `_forward(blob)` — прямой проход через сеть
- `_postprocess(image, outputs)` — извлечение детекций из выхода сети

### YOLOv3-tiny

**Препроцессинг:**
- Масштабирование до 416×416 пикселей
- Нормализация значений пикселей (деление на 255)
- Конвертация BGR → RGB
- Формирование blob через `cv2.dnn.blobFromImage`

**Постпроцессинг:**
- Декодирование центров и размеров bounding box
- Преобразование относительных координат в абсолютные
- Фильтрация по порогу уверенности
- Подавление немаксимумов (NMS) для каждого класса отдельно

### YOLOv4-tiny

**Препроцессинг:**
- Масштабирование до 608×608 пикселей
- Нормализация значений пикселей (деление на 255)
- Конвертация BGR → RGB
- Формирование blob через `cv2.dnn.blobFromImage`

**Постпроцессинг:**
- Декодирование центров и размеров bounding box
- Преобразование относительных координат в абсолютные
- Фильтрация по порогу уверенности
- Подавление немаксимумов (NMS) для каждого класса отдельно

### SSD MobileNet V2

**Препроцессинг:**
- Масштабирование до 300×300 пикселей
- Формирование blob через `cv2.dnn.blobFromImage`

**Постпроцессинг:**
- Извлечение нормализованных координат из выхода сети
- Преобразование в пиксельные координаты исходного изображения
- Обрезка выходящих за границы прямоугольников
- Применение NMS для устранения дубликатов

---

## Метрики качества

### Матрица ошибок

|               | Детекция: объект | Детекция: фон |
|---------------|------------------|---------------|
| **GT: объект** | TP               | FN            |
| **GT: фон**    | FP               | TN            |

### Формулы

**TPR (True Positive Rate / Recall):**
$$TPR = \frac{TP}{TP + FN}$$

**FDR (False Discovery Rate):**
$$FDR = \frac{FP}{TP + FP}$$

Сопоставление детекций с ground truth выполняется по метрике IoU с порогом 0.5.

---

## Результаты экспериментов

| Модель | TPR | FDR | TP | FP | FN |
|--------|-----|-----|----|----|-----|
| YOLOv4-tiny | **0.7914** | 0.0918 | 16059 | 1623 | 4233 |
| SSD MobileNet V2 | 0.4968 | **0.0195** | 10082 | 200 | 10210 |
| YOLOv3-tiny | 0.4673 | 0.5400 | 9482 | 11129 | 10810 |

**Выводы:**
- **YOLOv4-tiny** показывает лучший TPR (79.14%), обнаруживая наибольшее количество объектов
- **SSD MobileNet V2** имеет минимальный FDR (1.95%), но пропускает много объектов
- **YOLOv3-tiny** — самая быстрая модель, но имеет высокий FDR из-за большого количества ложных срабатываний

---

## Запуск приложения

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--images` | Путь к директории с изображениями | обязательный |
| `--ann` | Путь к файлу аннотаций | обязательный |
| `--model` | Выбор модели детектора | обязательный |
| `--conf` | Порог уверенности | 0.5 |
| `--nms` | Порог NMS | 0.4 |
| `--iou` | Порог IoU для метрик | 0.5 |
| `--show` | Отображать детекции | выкл |

### Примеры запуска

**YOLOv4-tiny (рекомендуется):**
```bash
python main.py --images data/imgs_MOV03478 --ann data/mov03478.txt --model yolo_v4_coco --show
```

**YOLOv3-tiny:**
```bash
python main.py --images data/imgs_MOV03478 --ann data/mov03478.txt --model yolo_v3_coco --show
```

**SSD MobileNet V2:**
```bash
python main.py --images data/imgs_MOV03478 --ann data/mov03478.txt --model ssd_mobilenet_v2_coco --show
```

**Запуск без визуализации (только метрики):**
```bash
python main.py --images data/imgs_MOV03478 --ann data/mov03478.txt --model yolo_v4_coco
```

### Управление при визуализации

- **Esc** — завершить работу
- **Крестик окна** — завершить работу
