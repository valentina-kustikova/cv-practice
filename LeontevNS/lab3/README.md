# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

## Описание проекта
Разработано приложение для классификации изображений достопримечательностей Нижнего Новгорода:
1. Нижегородский Кремль
2. Дворец труда
3. Архангельский собор

## Реализованные методы

### 1. Алгоритм "Мешок слов" (Bag of Words)
- Извлечение дескрипторов: SIFT, ORB, SURF
- Кластеризация: K-Means, MiniBatchKMeans
- Классификаторы: SVM, KNN, RandomForest, GradientBoosting

### 2. Нейросетевой классификатор
- Архитектура: MobileNetV2
- Метод: Трансферное обучение (Transfer Learning)
- Fine-tuning верхних слоев

## Структура проекта

```bash
landmark_classification/
├──data/                    # Данные изображений
├──splits/                  # Разделение на train/test
├──models/                  # Сохраненные модели
├──plots/                   # Графики и визуализации
├──bow.py                   # Реализация Bag of Words
├──NN.py                    # Реализация нейронной сети
├──load_data.py             # Загрузка и обработка данных
├──train_test_split.py      # Создание разделения данных
├──main.py                  # Основной скрипт
├──requirements.txt         # Зависимости
└──README.md               # Документация
```

## Установка и запуск

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```
2. Подготовка данных

Разместите изображения в директории data/ согласно структуре:
```bash
data/
├── NNSUDataset/
│   ├── 01_NizhnyNovgorodKremlin/
│   ├── 02_DvoretsTruda/
│   └── 03_ArkhangelskySobor/
└── ExtDataset/
    ├── 01_NizhnyNovgorodKremlin/
    ├── 02_DvoretsTruda/
    └── 03_ArkhangelskySobor/
```
3. Создание разделения данных
```bash
python train_test_split.py --data_dir data --output_dir splits --train_ratio 0.8
```
4. Обучение моделей

Bag of Words:

# Обучение с MiniBatchKMeans и SVM
```bash
python main.py --method BOW --type train \
  --train_file splits/train.txt \
  --clusters 300 \
  --clf_name SVC \
  --output models/bow_model.joblib
```

# С ORB дескрипторами и RandomForest
```bash
python main.py --method BOW --type train \
  --train_file splits/train.txt \
  --descriptor ORB \
  --clf_name RandomForest \
  --output models/bow_orb_rf.joblib
```

Нейронная сеть (MobileNetV2):

# Базовое обучение
```bash
python main.py --method NN --type train \
  --train_file splits/train.txt \
  --model_name MobileNetV2 \
  --epochs 20 \
  --batch_size_nn 16 \
  --output models/mobilenet_model.keras
```

5. Тестирование моделей

Bag of Words:
```bash
python main.py --method BOW --type test \
  --test_file splits/test.txt \
  --model models/bow_model.joblib \
  --verbose \
  --save_plots
```
Нейронная сеть:
```bash
python main.py --method NN --type test \
  --test_file splits/test.txt \
  --model models/efficientnet_model.keras \
  --verbose \
  --save_plots
```
6. Визуализация

Дескрипторов:
```bash
python main.py --method BOW --type visualize \
  --image data/NNSUDataset/01_NizhnyNovgorodKremlin/example.jpg \
  --descriptor SIFT \
  --save_plots
```
Предсказание для одного изображения:
```bash
python main.py --method NN --type predict \
  --image path/to/your/image.jpg \
  --model models/efficientnet_model.keras \
  --save_plots
```
7. Сравнение методов
```bash
python main.py --method compare --type evaluate \
  --test_file splits/test.txt \
  --model models/bow_model.joblib \
  --save_plots
```
Результаты экспериментов

Bag of Words (оптимальные параметры):

· Дескриптор: SIFT
· Кластеры: 300
· Кластеризация: MiniBatchKMeans
· Классификатор: SVC
· Точность: 96%

Нейронная сеть (MobileNetV2):

· Эпохи: 20 (базовое обучение) + 10 (fine-tuning)
· Точность: 98%

Особенности реализации

1. Bag of Words:

· Поддержка нескольких дескрипторов (SIFT, ORB, SURF)
· Разные методы кластеризации
· Несколько классификаторов на выбор
· Визуализация ключевых точек и гистограмм

2. Нейронная сеть:

· Использование MobileNetV2
· Автоматическая предобработка изображений
· Fine-tuning верхних слоев
· Callback-и для контроля обучения
· Визуализация истории обучения и confusion matrix

3. Общие возможности:

· Разделение на train/validation/test
· Сохранение и загрузка моделей
· Детальная статистика по классам
· Сохранение графиков и результатов