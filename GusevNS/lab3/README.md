# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

## Описание

Приложение для классификации изображений достопримечательностей Нижнего Новгорода (Нижегородский Кремль, Дворец труда, Архангельский собор) с использованием двух подходов: алгоритма "мешок слов" (Bag of Words) и нейросетевого классификатора.

## Реализованные алгоритмы

### 1. Bag of Words (BoW)

**Принцип работы:**
- Извлечение ключевых точек и дескрипторов из изображений (SIFT или ORB)
- Кластеризация дескрипторов с помощью K-Means для создания "визуального словаря"
- Построение гистограмм распределения визуальных слов для каждого изображения
- Классификация с использованием SVM (Support Vector Machine)

**Особенности реализации:**
- Поддержка двух типов дескрипторов: SIFT (более точный) и ORB (быстрее)
- Настраиваемое количество кластеров (по умолчанию 100)
- RBF ядро для SVM классификатора
- Нормализация гистограмм для инвариантности к количеству ключевых точек

### 2. Neural Network (Transfer Learning)

**Принцип работы:**
- Использование предобученной сети MobileNetV2 на ImageNet
- Заморозка первых слоев базовой модели
- Дообучение последних 20 слоев под новую задачу
- Добавление собственных полносвязных слоев с Dropout

**Особенности реализации:**
- Transfer learning для работы с малым объемом данных
- GlobalAveragePooling2D для уменьшения размерности
- Dropout (0.5) для предотвращения переобучения
- Adam оптимизатор с learning rate = 0.0001

## Установка зависимостей

```bash
pip install opencv-python opencv-contrib-python numpy scikit-learn tensorflow
```

## Использование

### Обучение и тестирование обоих алгоритмов

```bash
python lab3.py --mode both --algorithm both
```

### Обучение только Bag of Words

```bash
python lab3.py --mode train --algorithm bow --bow_clusters 150 --bow_descriptor sift
```

### Тестирование нейронной сети

```bash
python lab3.py --mode test --algorithm nn
```

### Тестирование готовых моделей (без обучения)
#### Тестирование Bag of Words модели
```bash
python lab3.py --mode test --algorithm bow
```

#### Тестирование нейронной сети
```bash
python lab3.py --mode test --algorithm nn
```

#### Тестирование обеих моделей сразу
```bash
python lab3.py --mode test --algorithm both
```

### Параметры запуска

- `--data_path` - путь к директории с данными (по умолчанию: `Data`)
- `--train_file` - файл с обучающей выборкой (по умолчанию: `Data/train.txt`)
- `--test_file` - файл с тестовой выборкой (по умолчанию: `Data/test.txt`)
- `--mode` - режим работы: `train`, `test`, `both` (по умолчанию: `both`)
- `--algorithm` - алгоритм: `bow`, `nn`, `both` (по умолчанию: `both`)
- `--bow_clusters` - количество кластеров для BoW (по умолчанию: 100)
- `--bow_descriptor` - тип дескриптора: `sift`, `orb` (по умолчанию: `sift`)
- `--nn_epochs` - количество эпох для нейросети (по умолчанию: 20)
- `--nn_batch` - размер батча (по умолчанию: 16)
- `--model_path` - путь для сохранения моделей (по умолчанию: `models`)

## Структура проекта

```
lab3/
├── Data/
│   ├── NNSUDataset/        # Фотографии студентов
│   ├── ExtDataset/         # Фотографии из интернета
│   ├── train.txt           # Обучающая выборка
│   └── test.txt            # Тестовая выборка
├── models/                 # Сохраненные модели
├── lab3.py                 # Основной код
└── README.md               # Этот файл
```

## Результаты

Приложение выводит следующие метрики:
- **Accuracy** - общая точность классификации
- **Precision, Recall, F1-score** - для каждого класса отдельно
- **Classification Report** - детальный отчет по всем классам

## Классы для классификации

1. **01_NizhnyNovgorodKremlin** - Нижегородский Кремль
2. **04_ArkhangelskCathedral** - Архангельский собор
3. **08_PalaceOfLabor** - Дворец труда
4. **77_airhockey** - Аэрохоккей

## Автор

Гусев Никита


## Результаты

### --mode test --algorithm nn
Точность (Accuracy): 0.9886

Отчет по классификации:
                          precision    recall  f1-score   support

01_NizhnyNovgorodKremlin       0.97      1.00      0.99        39
 04_ArkhangelskCathedral       1.00      0.95      0.97        20
        08_PalaceOfLabor       1.00      1.00      1.00        29

                accuracy                           0.99        88
               macro avg       0.99      0.98      0.99        88
            weighted avg       0.99      0.99      0.99        88

С новым паком изображений:
Точность (Accuracy): 0.9850

Отчет по классификации:
                          precision    recall  f1-score   support

01_NizhnyNovgorodKremlin       0.97      1.00      0.99        39
 04_ArkhangelskCathedral       1.00      1.00      1.00        20
        08_PalaceOfLabor       0.97      1.00      0.98        29
            77_airhockey       1.00      0.96      0.98        45

                accuracy                           0.98       133
               macro avg       0.99      0.99      0.99       133
            weighted avg       0.99      0.98      0.98       133



### -mode both --algorithm bow --bow_descriptor sift --bow_clusters 100
Точность (Accuracy): 0.9205

Отчет по классификации:
                          precision    recall  f1-score   support

01_NizhnyNovgorodKremlin       0.97      0.97      0.97        39
 04_ArkhangelskCathedral       0.94      0.75      0.83        20
        08_PalaceOfLabor       0.85      0.97      0.90        29

                accuracy                           0.92        88
               macro avg       0.92      0.90      0.90        88
            weighted avg       0.92      0.92      0.92        88

### --mode both --algorithm bow --bow_descriptor orb --bow_clusters 50
Точность (Accuracy): 0.8864

Отчет по классификации:
                          precision    recall  f1-score   support

01_NizhnyNovgorodKremlin       0.90      0.95      0.93        39
 04_ArkhangelskCathedral       0.89      0.80      0.84        20
        08_PalaceOfLabor       0.86      0.86      0.86        29

                accuracy                           0.89        88
               macro avg       0.88      0.87      0.88        88
            weighted avg       0.89      0.89      0.89        88

С новым паком изображений:
Точность (Accuracy): 0.8496

Отчет по классификации:
                          precision    recall  f1-score   support

01_NizhnyNovgorodKremlin       0.80      0.92      0.86        39
 04_ArkhangelskCathedral       0.78      0.70      0.74        20
        08_PalaceOfLabor       0.85      0.79      0.82        29
            77_airhockey       0.93      0.89      0.91        45

                accuracy                           0.85       133
               macro avg       0.84      0.83      0.83       133
            weighted avg       0.85      0.85      0.85       133

