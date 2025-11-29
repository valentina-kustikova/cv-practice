import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


def load_images_from_paths(file_paths, base_path, labels):
    """Загрузка изображений по путям из файла разбиения"""
    images = []
    image_labels = []
    failed_loads = []

    for file_path, label in zip(file_paths, labels):
        full_path = os.path.join(base_path, file_path)
        try:
            img = cv2.imread(full_path)
            if img is not None:
                images.append(img)
                image_labels.append(label)
            else:
                print(f"Не удалось загрузить изображение: {full_path}")
                failed_loads.append(full_path)
        except Exception as e:
            print(f"Ошибка загрузки {full_path}: {e}")
            failed_loads.append(full_path)

    if failed_loads:
        print(f"Не удалось загрузить {len(failed_loads)} изображений")

    return images, image_labels


def parse_split_file(split_file_path):
    """Парсинг файла разбиения и извлечение меток из путей"""
    with open(split_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    file_paths = []
    labels = []

    for line in lines:
        # Извлекаем метку из пути
        if '01_NizhnyNovgorodKremlin' in line:
            label = 'kremlin'
        elif '04_ArkhangelskCathedral' in line:
            label = 'sobor'
        elif '08_PalaceOfLabor' in line:
            label = 'palace'
        else:
            # Пропускаем неизвестные классы
            print(f"Неизвестный класс в пути: {line}")
            continue

        file_paths.append(line)
        labels.append(label)

    print(f"Найдено {len(file_paths)} валидных путей в файле {os.path.basename(split_file_path)}")
    return file_paths, labels


def find_all_images(data_path):
    """Находит все изображения в датасете"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_images = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                # Получаем относительный путь от data_path
                rel_path = os.path.relpath(full_path, data_path)
                all_images.append(rel_path)

    return all_images


def load_dataset(data_path, split_file_path):
    """Загрузка датасета на основе файла разбиения"""

    print("📁 Загрузка данных с правильным разбиением на train/test...")

    # Загрузка тренировочных данных из train.txt
    if os.path.exists(split_file_path):
        train_paths, train_labels = parse_split_file(split_file_path)
        train_images, train_labels = load_images_from_paths(train_paths, data_path, train_labels)
        print(f"✅ Загружено тренировочных изображений: {len(train_images)}")
    else:
        print(f"❌ Train file not found: {split_file_path}")
        return [], [], [], []

    # Находим ВСЕ изображения в датасете
    all_images = find_all_images(data_path)
    print(f"📊 Всего изображений в датасете: {len(all_images)}")

    # Получаем пути тренировочных изображений (относительные)
    train_relative_paths = [os.path.relpath(os.path.join(data_path, path), data_path)
                            for path in train_paths]

    # Тестовые данные = все изображения минус тренировочные
    test_paths = []
    test_labels = []

    for img_path in all_images:
        if img_path not in train_relative_paths:
            # Определяем метку для тестового изображения
            if '01_NizhnyNovgorodKremlin' in img_path:
                label = 'kremlin'
            elif '04_ArkhangelskCathedral' in img_path:
                label = 'sobor'
            elif '08_PalaceOfLabor' in img_path:
                label = 'palace'
            else:
                continue  # Пропускаем неизвестные классы

            test_paths.append(img_path)
            test_labels.append(label)

    # Загружаем тестовые изображения
    test_images, test_labels = load_images_from_paths(test_paths, data_path, test_labels)
    print(f"✅ Загружено тестовых изображений: {len(test_images)}")

    # Проверяем, что нет пересечений
    train_set = set(train_relative_paths)
    test_set = set(test_paths)
    intersection = train_set.intersection(test_set)

    if intersection:
        print(f"⚠️  Предупреждение: найдено {len(intersection)} пересекающихся изображений!")
    else:
        print("✅ Train и test наборы не пересекаются")

    return train_images, train_labels, test_images, test_labels


def evaluate_classifier(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)
    return accuracy, report


def visualize_keypoints(image, detector_name='SIFT'):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if detector_name == 'SIFT':
        detector = cv2.SIFT_create()
    elif detector_name == 'ORB':
        detector = cv2.ORB_create()
    else:
        detector = cv2.SIFT_create()

    keypoints = detector.detect(gray, None)
    img_with_kp = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_kp, len(keypoints)


def get_class_distribution(labels):
    """Получить распределение классов"""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def resize_images(images, target_size=(224, 224)):
    """Изменение размера изображений до целевого размера"""
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return resized_images