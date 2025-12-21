import argparse
import os

from feature_extractor import OpenCVFeatureExtractor
from visualizer import KeypointVisualizer


# ============================================================================
# CLI-скрипт для запуска визуализации ключевых точек.
# Поддерживает три режима:
# - single: визуализация одного детектора на изображении
# - compare: сравнение всех трех детекторов side-by-side
# - matches: поиск и отображение совпадающих точек между двумя изображениями
# Результаты сохраняются в директорию visualizations/.
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Визуализация ключевых точек для детекторов признаков'
    )

    parser.add_argument('--image', type=str, required=True,
                        help='Путь к изображению для визуализации')
    parser.add_argument('--detector', type=str, default='sift',
                        choices=['sift', 'orb', 'akaze'],
                        help='Тип детектора признаков')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'compare', 'matches'],
                        help='Режим визуализации')
    parser.add_argument('--image2', type=str, default=None,
                        help='Второе изображение (для режима matches)')
    parser.add_argument('--max_keypoints', type=int, default=None,
                        help='Максимальное количество ключевых точек для отображения')
    parser.add_argument('--max_matches', type=int, default=50,
                        help='Максимальное количество совпадений (для режима matches)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Директория для сохранения результатов')
    parser.add_argument('--no_show', action='store_true',
                        help='Не показывать окно с результатом')

    args = parser.parse_args()

    # Создание директории для результатов
    os.makedirs(args.output_dir, exist_ok=True)

    # Получение имени файла без расширения
    image_basename = os.path.splitext(os.path.basename(args.image))[0]

    if args.mode == 'single':
        # Визуализация ключевых точек для одного детектора
        print(f"Визуализация ключевых точек с детектором {args.detector.upper()}...")

        extractor = OpenCVFeatureExtractor(args.detector)
        visualizer = KeypointVisualizer(extractor)

        save_path = os.path.join(
            args.output_dir,
            f"{image_basename}_{args.detector}_keypoints.jpg"
        )

        visualizer.visualize_keypoints(
            args.image,
            save_path=save_path,
            show=not args.no_show,
            max_keypoints=args.max_keypoints
        )

    elif args.mode == 'compare':
        # Сравнение разных детекторов
        print("Сравнение детекторов SIFT, ORB и AKAZE...")

        extractors = [
            OpenCVFeatureExtractor('sift'),
            OpenCVFeatureExtractor('orb'),
            OpenCVFeatureExtractor('akaze')
        ]

        # Используем первый экстрактор для создания визуализатора
        visualizer = KeypointVisualizer(extractors[0])

        save_path = os.path.join(
            args.output_dir,
            f"{image_basename}_comparison.jpg"
        )

        visualizer.compare_detectors(
            args.image,
            extractors,
            save_path=save_path,
            show=not args.no_show,
            max_keypoints=args.max_keypoints
        )

    elif args.mode == 'matches':
        # Визуализация совпадений между двумя изображениями
        if args.image2 is None:
            print("Ошибка: для режима 'matches' необходимо указать --image2")
            return

        print(f"Визуализация совпадений с детектором {args.detector.upper()}...")

        extractor = OpenCVFeatureExtractor(args.detector)
        visualizer = KeypointVisualizer(extractor)

        image2_basename = os.path.splitext(os.path.basename(args.image2))[0]
        save_path = os.path.join(
            args.output_dir,
            f"{image_basename}_{image2_basename}_{args.detector}_matches.jpg"
        )

        visualizer.visualize_matches(
            args.image,
            args.image2,
            save_path=save_path,
            show=not args.no_show,
            max_matches=args.max_matches
        )

    print("Готово!")


if __name__ == '__main__':
    main()
