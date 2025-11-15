# Практическая работа №3. Классификация изображений с использованием библиотеки OpenCV

Поддерживается два алгоритма:  
1. «Мешок визуальных слов» (Bag of Visual Words, BoVW) с использованием OpenCV  
2. Нейросетевой классификатор на основе трансферного обучения (ResNet18, MobileNetV2)

# Описание модулей

## utils/dataset_loader.py

parse_class_from_path(path: str) -> int

Определяет метку класса по пути к файлу, анализируя имена директорий.
**Алгоритм:**
— Разбивает путь на компоненты 
— Ищет компонент, начинающийся с:
	+ "01_" → возвращает 0 (Кремль)
	+ "04_" → возвращает 1 (Архангельский собор)
	+ "08_" → возвращает 2 (Дворец труда)
	
load_split_lists(train_file: str, data_root: str) -> Tuple[List[str], List[str], List[int], List[int]]

Загружает тренировочную и тестовую выборки.

**Вход:**
 - train_file — путь к текстовому файлу со списком относительных путей (по одной строке на изображение)
 - data_root — корневая директория данных
 
**Алгоритм:**

 - Читает train_file, преобразует относительные пути в абсолютные (os.path.join(data_root, rel_path))
 - Для каждого пути извлекает метку через parse_class_from_path
 - Рекурсивно сканирует data_root и собирает все изображения (*.jpg, *.jpeg, *.png)
 - Формирует тестовую выборку как разность: все изображения \ тренировочные
 - Выводит статистику: число изображений в train/test, распределение по классам

## Класс BoVWClassifier

__init__(self, n_clusters: int = 100, detector_name: str = 'SIFT')

Инициализация модели.

**Алгоритм:**

 - Создаёт детектор и дескриптор (SIFT, ORB или AKAZE)
 - Инициализирует KMeans(n_clusters)
 - Создаёт pipeline: StandardScaler + SVC(kernel='rbf')
 - Устанавливает флаг is_fitted = False

_init_feature_extractor(self) -> Tuple[cv2.Feature2D, cv2.Feature2D]

Возвращает кортеж (detector, descriptor) в зависимости от detector_name:

 - 'SIFT': cv2.SIFT_create()
 - 'ORB': cv2.ORB_create(nfeatures=500)
 - 'AKAZE': cv2.AKAZE_create()

_extract_features(self, image_paths: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]

Извлекает дескрипторы из списка изображений.

**Алгоритм:**

Для каждого пути:
 - Загружает изображение через cv2.imread
 - Конвертирует в оттенки серого
 - Вызывает detectAndCompute
Объединяет все дескрипторы в единый массив all_descriptors (N × D)

fit(self, image_paths: List[str], labels: List[int])

Обучает модель BoVW.

**Алгоритм:**

 - Извлекает все дескрипторы через _extract_features
 - Обучает KMeans на объединённом наборе дескрипторов
 - Для каждого изображения строит гистограмму визуальных слов через _images_to_histograms
 - Обучает SVM на полученных гистограммах
 - Устанавливает is_fitted = True
 
_images_to_histograms(self, image_paths: List[str]) -> np.ndarray

Преобразует изображения в гистограммы визуальных слов.

**Алгоритм:**
Для каждого изображения:
 - Извлекает дескрипторы
 - Применяет kmeans.predict → получает индексы кластеров
 - Строит нормализованную гистограмму (np.histogram(..., density=True))
 - Возвращает матрицу M × n_clusters, где M — число изображений
 
 predict(self, image_paths: List[str]) -> np.ndarray
 
Выполняет предсказание меток.
 
**Алгоритм:**

 - Строит гистограммы через _images_to_histograms
 - Применяет classifier.predict

visualize_keypoints(self, image_path: str, save_to: str = "keypoints.jpg", max_points: int = 1000000)
Отрисовывает ключевые точки на изображении.

**Алгоритм:**

 - Загружает изображение
 - Извлекает ключевые точки через detector.detect
 - Ограничивает число точек до max_points
 - Рисует точки через cv2.drawKeypoints с флагом DRAW_RICH_KEYPOINTS
 - Добавляет текстовую подпись с именем детектора и числом точек
 - Сохраняет результат 
 
## Класс LandmarkDataset(Dataset)

__init__(self, image_paths: List[str], labels: List[int], transform=None)

Инициализация датасета.

**Алгоритм:**

 - Сохраняет пути и метки
 - Устанавливает transform (по умолчанию: Resize(224), ToTensor, Normalize)

__getitem__(self, idx: int) -> Tuple[torch.Tensor, int]

Загружает изображение по индексу:
 - Применяет transform
 - Возвращает (тензор, метка)

## Класс CNNClassifier

__init__(self, model_name: str = 'resnet18', num_classes: int = 3, device=None)
Инициализация модели.

**Алгоритм:**

 - Определяет устройство (cuda/cpu)
 - Вызывает _load_pretrained_model

_load_pretrained_model(self) -> torch.nn.Module

Загружает предобученную модель из локального файла.

**Алгоритм:**
 - Формирует путь: models/{model_name}.php ( fallback на .pth)
 - Создаёт архитектуру с 1000 выходами (weights=None)
 - Загружает state_dict из файла
 - Заменяет финальный слой на Linear(..., num_classes)
 - Возвращает модель
 
fit(self, train_paths, train_labels, val_paths=None, val_labels=None, batch_size=16, epochs=10, lr=1e-4)

Выполняет дообучение модели.

**Алгоритм:**
 - Создаёт DataLoader для train (и val, если задано)
 - Инициализирует CrossEntropyLoss, Adam, StepLR
 - В цикле по эпохам:
	 + Обучает модель на train
	 + Вычисляет loss и accuracy
	 + При наличии val — оценивает на валидации
	 + Обновляет learning rate
	 
_evaluate_loader(self, loader: DataLoader) -> float

Вычисляет accuracy на загрузчике в режиме eval.

predict(self, image_paths: List[str], batch_size: int = 32) -> List[int]

**Алгоритм:**
Выполняет инференс:
 - Создаёт датасет и загрузчик
 - В режиме no_grad получает предсказания
 - Возвращает список меток

## train.py

main()
Точка входа приложения.

**Алгоритм:**
 - Парсинг аргументов командной строки через argparse
 - Загрузка данных через load_split_lists
 - При наличии --vis_kp и algo=bovw — визуализация ключевых точек
 - В зависимости от --algo:
	 + bovw: создаёт BoVWClassifier, вызывает fit/predict
	 + cnn: создаёт CNNClassifier, вызывает fit/predict
 - Выводит точность на тестовой выборке