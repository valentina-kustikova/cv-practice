import cv2
import os
import numpy as np


def load_dataset_split(split_file):
    """Загрузка разбиения на выборки"""
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
    """Загрузка изображений на основе разбиения"""
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
    """Визуализация ключевых точек для отладки BOW"""
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