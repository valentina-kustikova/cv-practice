# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

## Цель работы

Разработка и сравнение алгоритмов компьютерного зрения для автоматической классификации фотографий архитектурных достопримечательностей Нижнего Новгорода.

## Реализованные методы

В рамках работы реализованы **два различных подхода** к задаче классификации:

1.  **Классический метод машинного обучения**: Bag of Words (BoW) с использованием SIFT-дескрипторов, кластеризации K-Means и классификатора SVM.
2.  **Современный подход глубокого обучения**: EfficientNetB0 на базе фреймворка TensorFlow/Keras с предобученными весами ImageNet.

### Распознаваемые объекты:
- **01_NizhnyNovgorodKremlin** - Нижегородский Кремль
- **04_ArkhangelskCathedral** - Архангельский собор  
- **08_PalaceOfLabor** - Дворец труда

---

## 1. Классический подход: Bag of Words (BoW)

### Принцип работы:

#### 1.1 Детектирование локальных признаков
- Применение алгоритма **SIFT** (Scale-Invariant Feature Transform) для поиска характерных точек на изображении.
- Формирование 128-мерных векторов-дескрипторов для каждой обнаруженной точки.
- SIFT устойчив к изменению масштаба, повороту и изменению освещения.

#### 1.2 Формирование визуального словаря
- Агрегация дескрипторов со всех изображений обучающей выборки.
- Применение алгоритма **K-means** для кластеризации дескрипторов.
- Центры кластеров образуют "визуальный словарь" (visual vocabulary).
- Размер словаря (количество кластеров) по умолчанию: 100.

#### 1.3 Векторизация изображений
- Для каждого изображения находятся дескрипторы.
- Каждый дескриптор сопоставляется с ближайшим словом из словаря.
- Строится гистограмма частот встречаемости визуальных слов.
- Гистограмма нормализуется (сумма элементов равна 1).

#### 1.4 Обучение классификатора
- Полученные гистограммы используются как векторы признаков.
- Обучается классификатор **SVM** (Support Vector Machine) с ядром RBF (Radial Basis Function).
- Используется оценка вероятностей (`probability=True`).

---

## 2. Современный подход: EfficientNetB0

### Принцип работы:

#### 2.1 Архитектурное решение
- Используется модель **EfficientNetB0** из библиотеки TensorFlow/Keras.
- Применяется техника **Transfer Learning** (перенос обучения): веса модели предобучены на масштабном датасете ImageNet.
- Базовая сеть (backbone) "замораживается" (веса не обновляются при обучении), что позволяет использовать извлеченные признаки без переобучения всей сети.

#### 2.2 Классификационная голова
- Поверх базовой сети добавляется новая "голова" для классификации под конкретную задачу:
    - `GlobalAveragePooling2D`: усреднение признаков.
    - `Dropout(0.2)`: регуляризация для предотвращения переобучения.
    - `Dense`: полносвязный слой с функцией активации `softmax` для получения вероятностей классов.

#### 2.3 Процесс обучения
- Обучается только добавленная классификационная голова.
- Оптимизатор: **Adam**.
- Функция потерь: `sparse_categorical_crossentropy`.
- Входные изображения приводятся к размеру 224x224.

### Конфигурация модели:
- **Базовая сеть**: EfficientNetB0 (заморожена)
- **Разрешение входа**: 224×224 пикселей
- **Размер пакета**: 16 (параметр `--batch_size`)
- **Скорость обучения**: 0.001 (параметр `--learning_rate`)
- **Число эпох**: 20 (параметр `--epochs`)
- **Оптимизатор**: Adam

---

## 3. Структура проекта

```
lab3/
├── classifier/                  # Пакет с реализацией классификаторов
│   ├── __init__.py              # Инициализация пакета
│   ├── base_classifier.py       # Абстрактный базовый класс
│   ├── bow_classifier.py        # Реализация BoW (SIFT + K-Means + SVM)
│   └── efficientnet_classifier.py # Реализация EfficientNet (TensorFlow)
├── NNClassification/            # Директория с данными (ExtDataset, NNSUDataset)
├── train.txt                    # Список файлов обучающей выборки
├── test.txt                     # Список файлов тестовой выборки
├── create_test_split.py         # Скрипт для генерации test.txt
├── bow_model/                   # Директория для сохранения модели BoW
│   └── bow_model.joblib         # Сохраненная модель BoW
├── efficientnet_model/          # Директория для сохранения модели EfficientNet
│   ├── efficientnet_model.keras # Сохраненные веса модели
│   └── metadata.json            # Метаданные модели
├── main.py                      # Основной скрипт запуска
├── README.md                    # Документация проекта
└── requirements.txt             # Список зависимостей
```

---

## 4. Описание классов и методов

### 4.1 BaseClassifier (`classifier/base_classifier.py`)
Базовый класс, определяющий интерфейс для всех классификаторов.
- `load_data(data_file, data_dir)`: Загружает пути к изображениям и метки классов из текстового файла.
- `load_image(image_path)`: Загружает изображение с диска.
- `evaluate(y_true, y_pred, target_names)`: Вычисляет метрики качества (precision, recall, f1-score).
- `train(train_paths, train_labels)`: Абстрактный метод обучения.
- `test(test_paths, test_labels)`: Абстрактный метод тестирования.

### 4.2 BOWClassifier (`classifier/bow_classifier.py`)
Реализация метода "Мешок слов".
- `extract_features(image)`: Извлекает SIFT-дескрипторы.
- `build_vocabulary(all_descriptors)`: Строит словарь визуальных слов с помощью K-Means.
- `descr_to_histogram(descriptors)`: Преобразует дескрипторы изображения в гистограмму.
- `train(...)`: Извлекает признаки, строит словарь, обучает SVM.
- `test(...)`: Предсказывает классы для тестовых изображений.
- `show_keypoints(...)`: Визуализирует ключевые точки SIFT на изображении.

### 4.3 EfficientNetClassifier (`classifier/efficientnet_classifier.py`)
Реализация нейросетевого классификатора.
- `create_model(n_classes)`: Создает архитектуру модели (EfficientNetB0 + Custom Head).
- `train(...)`: Подготавливает данные (tf.data.Dataset), выполняет обучение модели.
- `test(...)`: Выполняет предсказание на тестовых данных.
- `save_model()` / `load_model()`: Сохраняет и загружает модель и метаданные.

---

## 5. Команды для запуска

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Подготовка данных
Если файл `test.txt` отсутствует, его можно сгенерировать на основе `train.txt` и папки с данными:
```bash
python create_test_split.py --data_dir NNClassification --train_file train.txt --test_file test.txt
```

### Тестирование (с использованием готовых моделей)

**Тестирование BOW классификатора:**
```bash
python main.py --data_dir NNClassification --train_file train.txt --test_file test.txt --mode test --algorithm bow
```

**Тестирование EfficientNet классификатора:**
```bash
python main.py --data_dir NNClassification --train_file train.txt --test_file test.txt --mode test --algorithm efficientnet
```

**Тестирование обоих классификаторов (сравнение):**
```bash
python main.py --data_dir NNClassification --train_file train.txt --test_file test.txt --mode test --algorithm both
```

### Обучение и тестирование (полный цикл)

**Обучение и тестирование BOW классификатора:**
```bash
python main.py --data_dir NNClassification --train_file train.txt --test_file test.txt --mode both --algorithm bow
```

**Обучение и тестирование EfficientNet классификатора:**
```bash
python main.py --data_dir NNClassification --train_file train.txt --test_file test.txt --mode both --algorithm efficientnet --epochs 20
```

### Визуализация

**Визуализация SIFT-ключевых точек:**
```bash
python main.py --visualize "NNClassification/ExtDataset/01_NizhnyNovgorodKremlin/1e1f8c18-6a6b-41e5-bd48-abb3be5e388a.jpg" --visualize_output sift_visualization.jpg
```

---

## 6. Результаты экспериментов

### 6.1 Классический метод Bag of Words (BoW)

**Общая точность: 89.8%**

| Класс | Precision | Recall | F1-score | Support |
| :--- | :---: | :---: | :---: | :---: |
| 01_NizhnyNovgorodKremlin | 0.93 | 1.00 | 0.96 | 39 |
| 04_ArkhangelskCathedral | 0.87 | 0.65 | 0.74 | 20 |
| 08_PalaceOfLabor | 0.87 | 0.93 | 0.90 | 29 |
| **accuracy** | | | **0.90** | 88 |
| **macro avg** | 0.89 | 0.86 | 0.87 | 88 |
| **weighted avg** | 0.90 | 0.90 | 0.89 | 88 |

### 6.2 Современный подход EfficientNetB0

**Общая точность: 100.0%**

| Класс | Precision | Recall | F1-score | Support |
| :--- | :---: | :---: | :---: | :---: |
| 01_NizhnyNovgorodKremlin | 1.00 | 1.00 | 1.00 | 39 |
| 04_ArkhangelskCathedral | 1.00 | 1.00 | 1.00 | 20 |
| 08_PalaceOfLabor | 1.00 | 1.00 | 1.00 | 29 |
| **accuracy** | | | **1.00** | 88 |
| **macro avg** | 1.00 | 1.00 | 1.00 | 88 |
| **weighted avg** | 1.00 | 1.00 | 1.00 | 88 |

### 6.3 Сравнительный анализ

| Метод | Точность |
| :--- | :---: |
| **BoW + SVM** | 89.8% |
| **EfficientNetB0** | 100.0% |
