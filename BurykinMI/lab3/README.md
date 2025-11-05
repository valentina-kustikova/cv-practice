# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

## Цель работы
Разработать приложение для классификации изображений известных достопримечательностей Нижнего Новгорода (Нижегородский Кремль, Архангельский собор, Дворец труда). Приложение реализует алгоритм "мешок визуальных слов" (Bag of Words, BoW) с использованием различных детекторов и дескрипторов признаков. Набор данных загружается из директорий `ExtDataset` (фотографии из интернета) и `NNSUDataset` (фотографии студентов ИИТММ). Разбиение на тренировочную и тестовую выборки предоставляется в файле `train.txt`. Нейросетевой подход предусмотрен, но не реализован (TODO). 

Данные доступны по [ссылке на набор данных](https://cloud.unn.ru/s/2KsWFmaxzZf9mF5) (включая файл с источниками для `ExtDataset`). Разбиение на выборки — по [ссылке](https://cloud.unn.ru/s/ynKNZH9TxiwXqEb). Дополнительные изображения не добавлялись, использовался исходный набор.

---

## Структура проекта

```
BurykinMI/
└── lab3/
├── classifier.py                          # Основной скрипт с классом классификатора и main
├── README.md                              # Описание работы
├── data/
│   ├── train.txt                          # Файл разбиения на тренировочную и тестовую выборки
│   ├── ExtDataset/                        # Фотографии из интернета
│   │   ├── 01_NizhnyNovgorodKremlin/
│   │   ├── 04_ArkhangelskCathedral/
│   │   ├── 08_PalaceOfLabor/
│   │   └── references.csv                 # Ссылки на источники
│   └── NNSUDataset/                       # Фотографии студентов ИИТММ
│       ├── 01_NizhnyNovgorodKremlin/
│       ├── 04_ArkhangelskCathedral/
│       ├── 08_PalaceOfLabor/
├── models/
│   └── bow/
│       ├── akaze_clusters300.pkl          # Модель для AKAZE (300 кластеров)
│       ├── orb_clusters300.pkl            # Модель для ORB (300 кластеров)
│       ├── sift_clusters300.pkl           # Модель для SIFT (300 кластеров)
│       ├── akaze_clusters300_results.json # Результаты тестирования для AKAZE
│       ├── orb_clusters300_results.json   # Результаты тестирования для ORB
│       └── sift_clusters300_results.json  # Результаты тестирования для SIFT
```

---

## Описание реализованных алгоритмов

Приложение реализовано в объектно-ориентированном стиле с использованием класса `LandmarkClassifier`. Алгоритм основан на "мешке визуальных слов" (BoW): извлекаются признаки из изображений, строится словарь слов путем кластеризации, изображения представляются как гистограммы слов, и обучается классификатор SVM. Исследованы детекторы/дескрипторы: SIFT, ORB, AKAZE. Визуализация этапов не реализована, но возможна через OpenCV. Нейросетевой подход предусмотрен, но не реализован.

### Класс `LandmarkClassifier`

- **`__init__(self, algorithm='bow', detector_type='sift', n_clusters=300)`**:  
  Инициализация. Поддерживает алгоритмы 'bow' (реализован) и 'neural' (TODO). Инициализирует детектор (SIFT, ORB или AKAZE из OpenCV).

- **`extract_features(self, image_path)`**:  
  Извлечение признаков: чтение изображения, конвертация в grayscale, детекция ключевых точек и вычисление дескрипторов.

- **`load_dataset(self, data_dir, split_file)`**:  
  Загрузка датасета: чтение файлов из `ExtDataset` и `NNSUDataset`, разбиение по `train.txt`. Маппинг папок на классы: '01_NizhnyNovgorodKremlin' → 'kremlin', '04_ArkhangelskCathedral' → 'cathedral', '08_PalaceOfLabor' → 'palace'. Возвращает списки путей и меток для train/test. Выводит статистику распределения.

- **`build_vocabulary(self, image_paths)`**:  
  Построение словаря: извлечение дескрипторов из train-изображений, объединение, кластеризация с MiniBatchKMeans (sklearn).

- **`get_bow_features(self, image_path)`**:  
  Получение BoW-представления: извлечение дескрипторов, предсказание кластеров, построение нормализованной гистограммы.

- **`train(self, train_data, train_labels)`**:  
  Обучение: строит словарь, извлекает BoW-признаки, обучает SVM (rbf-ядро, C=10) из sklearn.

- **`predict(self, image_path)`**:  
  Предсказание класса для одного изображения на основе BoW и SVM.

- **`test(self, test_data, test_labels, results_path=None)`**:  
  Тестирование: предсказания для test, вычисление accuracy, classification_report, confusion_matrix (sklearn). Сохраняет результаты в JSON (если указан путь). Выводит метрики в консоль.

- **`save_model(self, filepath)`** и **`load_model(self, filepath)`**:  
  Сохранение/загрузка модели (kmeans, classifier, параметры) с pickle.

### Main функция

Использует argparse для парсинга параметров. Загружает данные, создает классификатор, выполняет train/test в зависимости от режима. Сохраняет модель в `models/` и результаты в JSON.

### Исследование и результаты

Исследованы детекторы SIFT, ORB, AKAZE с 300 кластерами. SVM выбран как классификатор (rbf-ядро для нелинейности). Результаты на тестовой выборке (88 изображений):

- **SIFT**: Accuracy = 0.9545  
  - Per-class: cathedral (P=1.0, R=0.8, F1=0.8889), kremlin (P=0.975, R=1.0, F1=0.9873), palace (P=0.9062, R=1.0, F1=0.9508)  
  - Confusion matrix: Минимальные ошибки, SIFT показывает наилучшее качество.

- **AKAZE**: Accuracy = 0.8977  
  - Per-class: cathedral (P=0.875, R=0.7, F1=0.7778), kremlin (P=0.8636, R=0.9744, F1=0.9157), palace (P=0.9643, R=0.931, F1=0.9474)  
  - Хуже SIFT, но лучше ORB.

- **ORB**: Accuracy = 0.7955  
  - Per-class: cathedral (P=0.8333, R=0.5, F1=0.625), kremlin (P=0.8182, R=0.9231, F1=0.8675), palace (P=0.75, R=0.8276, F1=0.7869)  
  - Наихудшее качество, больше ошибок в cathedral.

SIFT рекомендуется как лучший детектор. Датасет не расширялся, но код позволяет добавлять изображения в директории.

---

## Запуск программы

Скрипт `classifier.py` запускается с параметрами командной строки:
```bash
python classifier.py --data_dir <путь_к_данным> [опции]
```

### Примеры использования:

**Обучение и тестирование (BoW, SIFT, 300 кластеров):**
```bash
python classifier.py --data_dir data/ --mode train_test --algorithm bow --detector sift --n_clusters 300
```
- Обучает модель, тестирует, сохраняет модель в `models/bow/sift_clusters300.pkl` и результаты в `models/bow/sift_clusters300_results.json`.

**Только тестирование (загрузка модели):**
```bash
python classifier.py --data_dir data/ --mode test --algorithm bow --detector sift --n_clusters 300
```
- Загружает модель и тестирует.

**Другие детекторы:**
```bash
python classifier.py --data_dir data/ --mode train_test --algorithm bow --detector orb --n_clusters 300
python classifier.py --data_dir data/ --mode train_test --algorithm bow --detector akaze --n_clusters 300
```

**Нейросетевой режим (не реализован):**
```bash
python classifier.py --data_dir data/ --mode train_test --algorithm neural
```
- Вызовет NotImplementedError.

Если `--split_file` не указан, используется `data_dir/train.txt`. Директория моделей — `--models_dir` (по умолчанию 'models').

Вывод: Статистика датасета, прогресс обработки, метрики (accuracy, report, confusion matrix), пути сохранений.

---

## Требования к данным

- Директория `--data_dir` должна содержать `ExtDataset` и `NNSUDataset` с подпапками классов (`01_NizhnyNovgorodKremlin`, etc.).
- Файл разбиения (`train.txt`) — список относительных путей к train-изображениям.
- Изображения: JPG, JPEG, PNG, BMP.
- Если используются дополнительные фото, добавьте их в директории и обновите `train.txt`. Для сторонних — создайте файл ссылок (как в `ExtDataset`).

---

## Используемые библиотеки

- **OpenCV (cv2)**: Детекторы/дескрипторы (SIFT, ORB, AKAZE), чтение изображений.
- **NumPy**: Обработка массивов, гистограммы, нормализация.
- **Scikit-learn**: KMeans (MiniBatchKMeans), SVM (SVC), метрики (accuracy_score, classification_report, confusion_matrix).
- **Pickle**: Сохранение/загрузка моделей.
- **JSON, OS, Argparse**: Парсинг аргументов, работа с файлами, сохранение результатов.