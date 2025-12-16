#!/usr/bin/env python3
"""
Главное приложение для анализа видеопоследовательности.
Обеспечивает обнаружение транспортных средств с использованием нейронных сетей,
оценку качества и визуализацию результатов.
"""

import cv2
import os
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from detectors import SSDMobileNetDetector, YOLOv3Detector
from utils import DetectionEvaluator, load_annotation_data


class VideoSequenceAnalyzer:
    """Класс для анализа последовательности видеокадров"""

    def __init__(self):
        """Инициализация анализатора"""
        self.detector = None
        self.annotation_data = None
        self.evaluator = DetectionEvaluator()
        self.current_frame_number = 0
        self.is_paused = False

    def initialize_detector(self,
                            model_type: str,
                            config_paths: dict,
                            class_labels_path: str = None) -> bool:
        """
        Инициализирует детектор выбранного типа.

        Args:
            model_type: Тип модели ('ssd' или 'yolo')
            config_paths: Словарь с путями к конфигурациям
            class_labels_path: Путь к файлу с метками классов

        Returns:
            True если инициализация успешна, иначе False
        """
        if model_type == 'ssd':
            config = config_paths.get('ssd_config')
            weights = config_paths.get('ssd_weights')

            if not os.path.exists(weights):
                print(f"Ошибка: Файл весов SSD не найден: {weights}")
                return False

            try:
                self.detector = SSDMobileNetDetector(config, weights)
                print("Детектор SSD MobileNet инициализирован")
                return True
            except Exception as e:
                print(f"Ошибка инициализации SSD: {e}")
                return False

        elif model_type == 'yolo':
            config = config_paths.get('yolo_config')
            weights = config_paths.get('yolo_weights')
            classes = config_paths.get('yolo_classes')

            if not os.path.exists(weights):
                print(f"Ошибка: Файл весов YOLO не найден: {weights}")
                return False

            try:
                self.detector = YOLOv3Detector(config, weights, classes)
                print("Детектор YOLOv3 инициализирован")
                return True
            except Exception as e:
                print(f"Ошибка инициализации YOLO: {e}")
                return False
        else:
            print(f"Неизвестный тип модели: {model_type}")
            return False

    def load_annotations(self, annotation_file: str) -> bool:
        """
        Загружает данные аннотаций.

        Args:
            annotation_file: Путь к файлу аннотаций

        Returns:
            True если загрузка успешна, иначе False
        """
        self.annotation_data = load_annotation_data(annotation_file)
        return bool(self.annotation_data)

    def process_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Обрабатывает один кадр: детектирует объекты, обновляет метрики,
        рисует результаты.

        Args:
            frame: Входной кадр
            frame_id: Номер кадра

        Returns:
            Кадр с визуализацией
        """
        if self.detector is None:
            return frame

        # Получение аннотаций для текущего кадра
        frame_annotations = self.annotation_data.get(frame_id, [])

        # Детектирование объектов
        detected_objects = self.detector.detect_objects(frame)

        # Обновление метрик качества
        self.evaluator.add_detection_batch(detected_objects, frame_annotations)
        recall, false_discovery_rate = self.evaluator.get_performance_metrics()

        # Визуализация эталонных объектов (зеленый)
        for annotation in frame_annotations:
            class_name, x, y, w, h = annotation
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Текст снаружи рамки
            text_position = (x, y - 7 if y > 20 else y + h + 15)
            cv2.putText(frame,
                        f"vehicle: {class_name}",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1)

        # Визуализация обнаруженных объектов
        for detection in detected_objects:
            class_name, confidence, x, y, w, h = detection

            # Получение цвета для класса
            color = self.detector.get_class_color(class_name)

            # Отрисовка рамки
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Подготовка текста
            label_text = f"{class_name}: {confidence:.2f}"

            # Вычисление позиции текста (внутри рамки)
            text_y = y + 20 if y + 20 < y + h else max(20, y + h - 5)

            # Фон для текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            cv2.rectangle(frame,
                          (x, text_y - text_height - 5),
                          (x + text_width + 10, text_y + 5),
                          color,
                          -1)

            # Текст
            cv2.putText(frame,
                        label_text,
                        (x + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

        # Добавление информационной панели
        self._add_info_panel(frame, frame_id, recall, false_discovery_rate)

        return frame

    def _add_info_panel(self,
                        frame: np.ndarray,
                        frame_id: int,
                        recall: float,
                        fdr: float) -> None:
        """
        Добавляет информационную панель в верхнюю часть кадра.

        Args:
            frame: Кадр для рисования
            frame_id: Номер кадра
            recall: Значение полноты
            fdr: Значение ложных обнаружений
        """
        panel_height = 35
        cv2.rectangle(frame,
                      (0, 0),
                      (frame.shape[1], panel_height),
                      (40, 40, 40),
                      -1)

        # Текст статистики
        stats_text = (f"frame: {frame_id:06d} | "
                      f"recall: {recall:.3f} | "
                      f"false: {fdr:.3f}")

        cv2.putText(frame,
                    stats_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1)


    def toggle_pause(self) -> None:
        """Переключает состояние паузы"""
        self.is_paused = not self.is_paused
        status = "приостановлено" if self.is_paused else "возобновлено"
        print(f"Воспроизведение {status}")

    def get_final_metrics(self) -> tuple:
        """
        Возвращает итоговые метрики.

        Returns:
            Кортеж (полнота, ложные срабатывания)
        """
        return self.evaluator.get_performance_metrics()


def find_image_files(directory_path: str, pattern: str = "*.jpg") -> list:
    """
    Находит изображения в указанной директории.

    Args:
        directory_path: Путь к директории
        pattern: Шаблон поиска файлов

    Returns:
        Отсортированный список путей к файлам
    """
    if not os.path.isdir(directory_path):
        print(f"Директория не найдена: {directory_path}")
        return []

    search_pattern = os.path.join(directory_path, pattern)
    image_files = sorted(glob.glob(search_pattern))

    if not image_files:
        # Попробуем другие расширения
        for ext in ["*.png", "*.jpeg", "*.bmp"]:
            image_files = sorted(glob.glob(os.path.join(directory_path, ext)))
            if image_files:
                break

    return image_files


def extract_frame_number(file_path: str) -> int:
    """
    Извлекает номер кадра из имени файла.

    Args:
        file_path: Путь к файлу

    Returns:
        Номер кадра или -1 при ошибке
    """
    try:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Удаляем ведущие нули
        return int(name_without_ext.lstrip('0') or 0)
    except ValueError:
        return -1


def main():
    """Главная функция приложения"""
    parser = argparse.ArgumentParser(
        description="Анализ видеопоследовательности с обнаружением транспортных средств",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python video_analysis_app.py --model ssd
  python video_analysis_app.py --model yolo --frames_dir data/sequence1
        """
    )

    parser.add_argument("--model",
                        type=str,
                        default="ssd",
                        choices=["ssd", "yolo"],
                        help="Тип модели детектора (по умолчанию: ssd)")

    parser.add_argument("--frames_dir",
                        type=str,
                        default="data/MOV03478",
                        help="Директория с кадрами видео")

    parser.add_argument("--annotation_file",
                        type=str,
                        default="data/MOV03478/mov03478.txt",
                        help="Файл с аннотациями (разметкой)")

    # Параметры моделей
    parser.add_argument("--ssd_config",
                        default="models/ssd/MobileNetSSD_deploy.prototxt.txt",
                        help="Конфигурация SSD модели")

    parser.add_argument("--ssd_weights",
                        default="models/ssd/MobileNetSSD_deploy.caffemodel",
                        help="Веса SSD модели")

    parser.add_argument("--yolo_config",
                        default="models/yolo/yolov3.cfg",
                        help="Конфигурация YOLO модели")

    parser.add_argument("--yolo_weights",
                        default="models/yolo/yolov3.weights",
                        help="Веса YOLO модели")

    parser.add_argument("--yolo_classes",
                        default="models/yolo/coco.names",
                        help="Файл с классами YOLO")

    args = parser.parse_args()

    # Проверка существования директорий
    if not os.path.exists(args.frames_dir):
        print(f"Ошибка: Директория с кадрами не найдена: {args.frames_dir}")
        return

    # Инициализация анализатора
    analyzer = VideoSequenceAnalyzer()

    # Настройка путей к моделям
    model_paths = {
        'ssd_config': args.ssd_config,
        'ssd_weights': args.ssd_weights,
        'yolo_config': args.yolo_config,
        'yolo_weights': args.yolo_weights,
        'yolo_classes': args.yolo_classes
    }

    # Инициализация детектора
    print(f"\n{'=' * 60}")
    print(f"Запуск анализа с моделью: {args.model.upper()}")
    print(f"{'=' * 60}\n")

    if not analyzer.initialize_detector(args.model, model_paths):
        print("Не удалось инициализировать детектор. Завершение работы.")
        return

    # Загрузка аннотаций
    print("Загрузка данных аннотаций...")
    if not analyzer.load_annotations(args.annotation_file):
        print("Предупреждение: Аннотации не загружены или файл пуст")

    # Поиск файлов изображений
    print("Поиск кадров видео...")
    image_files = find_image_files(args.frames_dir)

    if not image_files:
        print(f"Кадры не найдены в директории: {args.frames_dir}")
        return

    print(f"Найдено кадров: {len(image_files)}")
    print("\nЗапуск анализа... (ESC - выход, ПРОБЕЛ - пауза)\n")

    # Основной цикл обработки
    window_name = "Cars"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for image_path in image_files:
        # Загрузка кадра
        frame_data = cv2.imread(image_path)
        if frame_data is None:
            print(f"Не удалось загрузить кадр: {image_path}")
            continue

        # Извлечение номера кадра
        frame_number = extract_frame_number(image_path)
        if frame_number < 0:
            print(f"Неверный формат имени файла: {image_path}")
            continue

        # Обработка кадра
        processed_frame = analyzer.process_frame(frame_data, frame_number)

        # Отображение результата
        cv2.imshow(window_name, processed_frame)

        # Обработка клавиш
        while analyzer.is_paused:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                print("\nАнализ прерван пользователем")
                return
            elif key == 32:  # Пробел
                analyzer.toggle_pause()
                break
            elif key == ord('q'):  # Дополнительная клавиша выхода
                cv2.destroyAllWindows()
                return

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # Пробел
            analyzer.toggle_pause()

    # Завершение работы
    cv2.destroyAllWindows()

    # Вывод итоговых результатов
    final_recall, final_fdr = analyzer.get_final_metrics()

    print(f"\n{'=' * 60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА")
    print(f"{'=' * 60}")
    print(f"Модель:               {args.model.upper()}")
    print(f"Обработано кадров:    {len(image_files)}")
    print(f"Итоговая полнота:     {final_recall:.4f}")
    print(f"Ложные срабатывания:  {final_fdr:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()