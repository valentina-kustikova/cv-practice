# Практическая работа №2. Детектирование объектов на изображениях с использованием библиотеки OpenCV

## Структура
- `detectors/` - иерархия классов детекторов на базе OpenCV DNN
- `utils/` - загрузка датасета, визуализация, расчёт метрик
- `app.py` - демонстрационное приложение

## Использованные модели

### 1. YOLOv4-tiny (Darknet)

**Предобработка:**
- Изменение размера изображения до 416×416
- Нормализация значений пикселей делением на 255 (диапазон [0, 1])
- Перестановка каналов BGR → RGB (`swapRB=True`)
- Формирование blob через `cv2.dnn.blobFromImage`

**Постобработка:**
- Получение выходов со всех выходных слоёв сети (`getUnconnectedOutLayersNames`)
- Для каждого детектирования извлечение координат центра, ширины, высоты и вектора вероятностей классов
- Выбор класса с максимальной вероятностью, фильтрация по порогу уверенности
- Фильтрация по транспортным классам (car, truck, bus, motorbike)
- Пересчёт координат из относительных в абсолютные
- Применение Non-Maximum Suppression (`cv2.dnn.NMSBoxes`)

### 2. MobileNet-SSD (Caffe, VOC)

**Предобработка:**
- Изменение размера до 300×300
- Масштабирование пикселей: умножение на 0.007843
- Вычитание среднего значения 127.5
- Формирование blob через `cv2.dnn.blobFromImage`

**Постобработка:**
- Прямой проход через сеть (`net.forward()`)
- Выход имеет форму [1, 1, N, 7], где каждая детекция содержит: batch_id, class_id, confidence, x1, y1, x2, y2
- Фильтрация по порогу уверенности и транспортным классам VOC (car, bus, motorbike)
- Пересчёт относительных координат в абсолютные

### 3. SSD Inception V2 (TensorFlow, COCO)

**Предобработка:**
- Использование `cv2.dnn_DetectionModel` с автоматической предобработкой
- Размер входа 300×300
- Масштаб 1.0, среднее (0, 0, 0)
- Перестановка каналов BGR → RGB

**Постобработка:**
- Вызов `net.detect()` возвращает классы, уверенности и боксы
- Коррекция индексов классов COCO (сдвиг на 1)
- Фильтрация по транспортным классам (car, truck, bus, motorcycle)

## Расчёт метрик

- **IoU** (Intersection over Union) используется для сопоставления детекций с разметкой
- **TPR** (True Positive Rate) = TP / (TP + FN)
- **FDR** (False Discovery Rate) = FP / (FP + TP)
- Порог IoU задаётся параметром `--iou` (по умолчанию 0.5)

## Запуск

```bash
python app.py --model yolov4_tiny --images Data/imgs_MOV03478 --annotations Data/mov03478.txt --display --limit 100
```

## Запуск с графикой
```bash
python app.py --model yolov4_tiny --display --limit 100
```

## Все кадры
### yolov4 tiny
```bash
python app.py --model yolov4_tiny
```
Processed frames: 3456
True positives: 18150
False positives: 4143
False negatives: 2142
TPR: 0.894
FDR: 0.186


### mobilenet ssd
```bash
python app.py --model mobilenet_ssd
```
Processed frames: 3456
True positives: 12046
False positives: 445
False negatives: 8246
TPR: 0.594
FDR: 0.036

score_threshold 0.4 -> 0.2:
Processed frames: 3456
True positives: 13452
False positives: 681
False negatives: 6840
TPR: 0.663
FDR: 0.048



### ssd inception
```bash
python app.py --model ssd_inception
```
Processed frames: 3456
True positives: 12225
False positives: 1189
False negatives: 8067
TPR: 0.602
FDR: 0.089

score_threshold 0.4 -> 0.2:
Processed frames: 3456
True positives: 16975
False positives: 5449
False negatives: 3317
TPR: 0.837
FDR: 0.243



**Параметры:**
- `--model` - выбор модели: `yolov4_tiny`, `mobilenet_ssd`, `ssd_inception`
- `--images` - путь к папке с изображениями
- `--annotations` - путь к файлу разметки
- `--display` - включить визуализацию (Esc для выхода)
- `--limit` - ограничить число обрабатываемых кадров (3456 - максимум)
- `--iou` - порог IoU для сопоставления

## Требования
- Python 3.9+
- `pip install opencv-python numpy`
