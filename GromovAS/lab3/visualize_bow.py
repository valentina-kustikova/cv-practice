import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset_split, load_images_from_split, visualize_keypoints_detailed


def visualize_bow_features(data_dir, split_file, output_dir, num_images=5, detector='SIFT'):
    # Визуализация ключевых точек для нескольких изображений

    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем данные
    files, labels = load_dataset_split(split_file)
    images_data = load_images_from_split(data_dir, files, labels)

    print(f"Визуализация ключевых точек для {min(num_images, len(images_data))} изображений...")
    print(f"Детектор: {detector}")

    results = []

    for i, (image, label) in enumerate(images_data[:num_images]):
        print(f"Обработка изображения {i + 1}: {label}")

        # Визуализация ключевых точек
        save_path = os.path.join(output_dir, f'kp_{i}_{label}_{detector}.jpg')
        image_with_kp, num_kp, descriptors = visualize_keypoints_detailed(
            image, detector, save_path
        )

        results.append({
            'original_image': image,
            'image_with_kp': image_with_kp,
            'num_keypoints': num_kp,
            'label': label,
            'descriptors_shape': descriptors.shape if descriptors is not None else None,
            'save_path': save_path
        })

        print(f"  - Найдено ключевых точек: {num_kp}")
        if descriptors is not None:
            print(f"  - Размер дескрипторов: {descriptors.shape}")

    return results


def create_comparison_plot(results, output_path):
    # Создание сравнительного графика
    fig, axes = plt.subplots(2, len(results), figsize=(15, 8))

    if len(results) == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, result in enumerate(results):
        # Оригинальное изображение
        axes[0][i].imshow(result['original_image'])
        axes[0][i].set_title(f'Original: {result["label"]}')
        axes[0][i].axis('off')

        # Изображение с ключевыми точками
        axes[1][i].imshow(result['image_with_kp'])
        axes[1][i].set_title(f'Keypoints: {result["num_keypoints"]}')
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # Параметры визуализации
    data_dir = './data'
    split_file = './data/train_fixed.txt'  # или test_fixed.txt
    output_dir = './visualization_results'
    detector = 'SIFT'  # или 'ORB'
    num_images = 3

    print("=== ВИЗУАЛИЗАЦИЯ КЛЮЧЕВЫХ ТОЧЕК BOW ===")

    # Визуализация ключевых точек
    results = visualize_bow_features(data_dir, split_file, output_dir, num_images, detector)

    # Создание сравнительного графика
    comparison_path = os.path.join(output_dir, 'comparison.jpg')
    create_comparison_plot(results, comparison_path)

    print(f"\nРезультаты сохранены в: {output_dir}")
    print(f"Сравнительный график: {comparison_path}")

    # Статистика по ключевым точкам
    print("\n=== СТАТИСТИКА ===")
    for result in results:
        print(f"{result['label']}: {result['num_keypoints']} точек")


if __name__ == "__main__":
    main()