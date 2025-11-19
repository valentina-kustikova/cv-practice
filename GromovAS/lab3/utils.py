import cv2
import os
import numpy as np


def load_dataset_split(split_file):
    # Загрузка разбиения на выборки
    files = []
    labels = []

    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    files.append(parts[0])
                    labels.append(parts[1])

    return files, labels


def load_images_from_split(data_dir, files, labels):
    # Загрузка изображений на основе разбиения
    images_data = []

    for file_path, label in zip(files, labels):
        full_path = os.path.join(data_dir, file_path)

        if os.path.exists(full_path):
            # Загрузка изображения
            image = cv2.imread(full_path)
            if image is not None:
                # Конвертация в RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_data.append((image_rgb, label))
            else:
                print(f"Ошибка загрузки: {full_path}")
        else:
            print(f"Файл не найден: {full_path}")

    return images_data


def visualize_keypoints(image, detector_type='SIFT'):
    # Визуализация ключевых точек для отладки BOW
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create()
    elif detector_type == 'ORB':
        detector = cv2.ORB_create()

    keypoints, descriptors = detector.detectAndCompute(image, None)
    image_with_kp = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return image_with_kp, len(keypoints)


def visualize_keypoints_detailed(image, detector_type='SIFT', save_path=None):
    # Детальная визуализация ключевых точек с информацией
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create()
    elif detector_type == 'ORB':
        detector = cv2.ORB_create()
    else:
        detector = cv2.SIFT_create()  # fallback

    # Детектирование ключевых точек и дескрипторов
    keypoints, descriptors = detector.detectAndCompute(image, None)

    # Визуализация ключевых точек
    image_with_kp = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)  # Зеленый цвет для точек
    )

    # Добавление информации о количестве точек
    cv2.putText(image_with_kp,
                f'Keypoints: {len(keypoints)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.putText(image_with_kp,
                f'Detector: {detector_type}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    # Сохранение если указан путь
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_with_kp, cv2.COLOR_RGB2BGR))

    return image_with_kp, len(keypoints), descriptors