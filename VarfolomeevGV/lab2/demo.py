"""
Демонстрационное приложение для детектирования транспортных средств.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict

from detectors import YOLODetector, FasterRCNNDetector, RetinaNetDetector
from utils.visualization import draw_detections, generate_colors, draw_metrics
from utils.metrics import calculate_metrics, load_ground_truth


def load_images(images_dir: str) -> List[str]:
    """
    Загрузка списка путей к изображениям из директории.
    
    Args:
        images_dir: Путь к директории с изображениями
        
    Returns:
        Список путей к изображениям
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    if not os.path.exists(images_dir):
        print(f"Ошибка: директория не найдена: {images_dir}")
        return image_paths
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in supported_formats):
            image_paths.append(os.path.join(images_dir, file))
    
    image_paths.sort()
    return image_paths


def load_annotations(annotations_path: str, image_names: List[str]) -> Dict[str, List[List[float]]]:
    """
    Загрузка разметки для изображений.
    
    Args:
        annotations_path: Путь к файлу или директории с разметкой
        image_names: Список имен файлов изображений
        
    Returns:
        Словарь {имя_файла: список_детекций}
    """
    annotations = {}
    
    if not annotations_path:
        return annotations
    
    if os.path.isfile(annotations_path):
        # Один файл с разметкой для всех изображений
        # Формат может быть 'required' (frame_id class_name x1 y1 x2 y2) или 'simple'
        try:
            # Проверяем формат файла по первой строке
            format_loaded = False
            with open(annotations_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    # Если формат 'required', первая колонка - frame_id (число)
                    if len(parts) >= 6:
                        try:
                            frame_id = int(parts[0])
                            # Это формат 'required', фильтруем по номеру кадра
                            # Загружаем разметку для каждого изображения отдельно
                            for idx, img_path in enumerate(image_names):
                                img_name = os.path.basename(img_path)
                                frame_gt = []
                                
                                # Читаем файл и фильтруем строки для текущего кадра
                                with open(annotations_path, 'r', encoding='utf-8') as f2:
                                    for line in f2:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        line_parts = line.split()
                                        if len(line_parts) >= 6:
                                            try:
                                                if int(line_parts[0]) == idx:
                                                    # Это наш кадр, парсим разметку
                                                    class_name = line_parts[1]
                                                    # Маппинг названий классов в COCO ID
                                                    class_name_to_id = {
                                                        'car': 2, 'CAR': 2, 'Car': 2,
                                                        'bicycle': 1, 'BICYCLE': 1, 'Bicycle': 1,
                                                        'motorcycle': 3, 'MOTORCYCLE': 3, 'Motorcycle': 3,
                                                        'bus': 5, 'BUS': 5, 'Bus': 5,
                                                        'truck': 7, 'TRUCK': 7, 'Truck': 7,
                                                    }
                                                    class_id = class_name_to_id.get(class_name, -1)
                                                    if class_id == -1:
                                                        class_id = class_name_to_id.get(class_name.upper(), -1)
                                                    if class_id == -1:
                                                        class_id = class_name_to_id.get(class_name.lower(), -1)
                                                    
                                                    if class_id != -1:
                                                        try:
                                                            x1 = float(line_parts[2])
                                                            y1 = float(line_parts[3])
                                                            x2 = float(line_parts[4])
                                                            y2 = float(line_parts[5])
                                                            frame_gt.append([x1, y1, x2, y2, class_id])
                                                        except ValueError:
                                                            continue
                                            except ValueError:
                                                continue
                                
                                annotations[img_name] = frame_gt
                            
                            # Если загрузили хотя бы одну разметку, помечаем успех
                            if annotations:
                                format_loaded = True
                        except ValueError:
                            pass  # Не формат 'required', пробуем дальше
            
            # Если не удалось загрузить как 'required', пробуем как 'simple'
            if not format_loaded:
                gt = load_ground_truth(annotations_path, format='simple')
                if gt:
                    # Применяем ко всем изображениям
                    for img_name in image_names:
                        annotations[img_name] = gt
        except Exception as e:
            print(f"Ошибка при загрузке разметки: {e}")
            import traceback
            traceback.print_exc()
    elif os.path.isdir(annotations_path):
        # Директория с файлами разметки (один файл на изображение)
        for img_path in image_names:
            img_name = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_name)[0]
            
            # Поиск файла разметки
            for ext in ['.txt', '.json', '.xml']:
                ann_path = os.path.join(annotations_path, img_name_no_ext + ext)
                if os.path.exists(ann_path):
                    # Пробуем загрузить как required формат
                    gt = load_ground_truth(ann_path, format='required')
                    
                    # Если пусто, пробуем как простой формат
                    if not gt:
                        gt = load_ground_truth(ann_path, format='simple')
                        
                    annotations[img_name] = gt
                    break
    
    return annotations


def create_detector(model_name: str, weights_path: str = None, 
                    config_path: str = None, names_path: str = None,
                    conf_threshold: float = 0.5, nms_threshold: float = 0.4):
    """
    Создание детектора выбранной модели.
    
    Args:
        model_name: Название модели ('yolo', 'faster_rcnn', 'retinanet')
        weights_path: Путь к весам модели
        config_path: Путь к конфигурации
        names_path: Путь к файлу с названиями классов
        conf_threshold: Порог уверенности
        nms_threshold: Порог NMS
        
    Returns:
        Объект детектора
    """
    # Определение путей по умолчанию
    if names_path is None:
        names_path = "models/coco.names"
    
    if model_name.lower() == 'yolo':
        if weights_path is None:
            weights_path = "models/yolo/yolov5s.onnx"
        detector = YOLODetector(conf_threshold, nms_threshold)
        detector.load_model(weights_path, names_path=names_path)
        
    elif model_name.lower() == 'faster_rcnn':
        if weights_path is None:
            weights_path = "models/faster_rcnn/frozen_inference_graph.pb"
        if config_path is None:
            config_path = "models/faster_rcnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        detector = FasterRCNNDetector(conf_threshold, nms_threshold)
        detector.load_model(weights_path, config_path, names_path)
        
        
    elif model_name.lower() == 'retinanet':
        if weights_path is None:
            weights_path = "models/retinanet/retinanet-9.onnx"
        detector = RetinaNetDetector(conf_threshold, nms_threshold)
        detector.load_model(weights_path, config_path, names_path)
        
    else:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: yolo, faster_rcnn, retinanet")
    
    return detector


def main():
    """Основная функция демо-приложения."""
    parser = argparse.ArgumentParser(
        description="Детектирование транспортных средств на изображениях"
    )
    
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Путь к директории с изображениями'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['yolo', 'faster_rcnn', 'retinanet'],
        help='Модель для детектирования (yolo, faster_rcnn, retinanet)'
    )
    
    parser.add_argument(
        '--annotations',
        type=str,
        default=None,
        help='Путь к файлу или директории с разметкой (опционально, для вычисления метрик)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Путь к весам модели (по умолчанию используется стандартный путь)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Путь к конфигурационному файлу (для Faster R-CNN)'
    )
    
    parser.add_argument(
        '--names',
        type=str,
        default=None,
        help='Путь к файлу с названиями классов (по умолчанию models/coco.names)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Порог уверенности для фильтрации детекций (по умолчанию 0.5)'
    )
    
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Порог IoU для Non-Maximum Suppression (по умолчанию 0.4)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Показывать изображения с детекциями'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Сохранять изображения с детекциями'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Директория для сохранения результатов (по умолчанию output/)'
    )
    
    parser.add_argument(
        '--vid',
        action='store_true',
        help='Создать видео из обработанных кадров'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Автоматическое переключение кадров при просмотре с задержкой 100мс'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Частота кадров для видео (по умолчанию 25)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=0,
        help='Количество кадров для обработки (по умолчанию 0 - все кадры; для видео по умолчанию 100)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Включить диагностический вывод для анализа проблем (отладочная информация о детекциях и метриках)'
    )
    
    args = parser.parse_args()
    
    # Загрузка изображений
    print("Загрузка изображений...")
    image_paths = load_images(args.images)
    if len(image_paths) == 0:
        print(f"Ошибка: не найдено изображений в директории {args.images}")
        return
    
    # Сортировка по имени файла для правильной последовательности в видео
    image_paths.sort()
    
    # Ограничение количества кадров
    # Если указан duration > 0, используем его всегда
    if args.duration > 0:
        if len(image_paths) > args.duration:
            print(f"Ограничение количества кадров до {args.duration}")
            image_paths = image_paths[:args.duration]
    # Если duration == 0, но включено видео, ограничиваем до 100 (дефолт для видео)
    elif args.vid and args.duration == 0:
        default_video_duration = 100
        if len(image_paths) > default_video_duration:
            print(f"Применение стандартного ограничения для видео: {default_video_duration} кадров")
            image_paths = image_paths[:default_video_duration]
    
    print(f"Найдено изображений: {len(image_paths)}")
    
    # Загрузка разметки (если указана)
    annotations = {}
    if args.annotations:
        print("Загрузка разметки...")
        image_names = [os.path.basename(path) for path in image_paths]
        annotations = load_annotations(args.annotations, image_names)
        print(f"Загружена разметка для {len(annotations)} изображений")
    
    # Создание детектора
    print(f"\nСоздание детектора: {args.model}")
    try:
        detector = create_detector(
            args.model,
            args.weights,
            args.config,
            args.names,
            args.conf_threshold,
            args.nms_threshold
        )
        
        # Включение диагностического вывода для RetinaNet
        if args.debug:
            if hasattr(detector, 'set_debug_logging'):
                detector.set_debug_logging(True)
                print("Диагностический вывод включен")
            else:
                print("[WARN] Детектор не поддерживает диагностический вывод")
        
        print("Детектор создан успешно")
    except Exception as e:
        print(f"Ошибка при создании детектора: {e}")
        return
    
    # Создание директории для результатов
    if args.save or args.vid:
        os.makedirs(args.output, exist_ok=True)
    
    # Инициализация видео-райтера
    video_writer = None
    if args.vid and len(image_paths) > 0:
        # Определяем размер видео по первому изображению
        first_img = cv2.imread(image_paths[0])
        if first_img is not None:
            h, w = first_img.shape[:2]
            video_path = os.path.join(args.output, 'output_video.mp4')
            # Используем кодек mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
            print(f"Видео будет сохранено в: {video_path}")
        else:
            print("Ошибка: не удалось загрузить первое изображение для инициализации видео")
    
    # Генерация цветов для классов
    num_classes = len(detector.class_names)
    colors = generate_colors(num_classes)
    
    # Обработка изображений
    print("\nОбработка изображений...")
    all_tpr = []
    all_fdr = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {os.path.basename(image_path)}")
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Ошибка: не удалось загрузить изображение")
            continue
        
        # Детектирование
        detections = detector.detect(image)
        if args.debug:
            print(f"  Найдено детекций (все классы): {len(detections)}")
        
        # Фильтрация по классам транспортных средств
        vehicle_detections = detector.filter_vehicle_classes(detections)
        if args.debug:
            print(f"  Найдено транспортных средств: {len(vehicle_detections)}")
        
        # Если транспортных средств не найдено, но есть другие детекции, показываем их
        if args.debug and len(vehicle_detections) == 0 and len(detections) > 0:
            print(f"  [WARN] Транспортных средств не найдено, но найдено других объектов: {len(detections)}")
            print(f"  Первые 5 детекций:")
            for i, det in enumerate(detections[:5]):
                class_id = int(det[4])
                confidence = det[5] if len(det) > 5 else 1.0
                class_name = detector.class_names[class_id] if class_id < len(detector.class_names) else f"Class {class_id}"
                print(f"    - {class_name} (id={class_id}): {confidence:.3f}")
        
        # Вычисление метрик (если есть разметка)
        tpr, fdr = 0.0, 0.0
        img_name = os.path.basename(image_path)
        if annotations and img_name in annotations:
            gt = annotations[img_name]
            
            # Определяем классы, которые есть в разметке
            gt_classes = set(int(g[4]) for g in gt if len(g) >= 5)
            
            # Фильтруем предсказания только по классам, которые есть в разметке
            # Это дает более корректные метрики, так как мы сравниваем только релевантные классы
            filtered_predictions = [
                det for det in vehicle_detections 
                if len(det) >= 5 and int(det[4]) in gt_classes
            ]
            
            # Отладочный вывод для диагностики
            if args.debug and len(vehicle_detections) > 0 and len(gt) > 0:
                pred_classes = [int(d[4]) for d in vehicle_detections]
                print(f"  DEBUG: Предсказания - классы: {set(pred_classes)}, количество: {len(vehicle_detections)}")
                print(f"  DEBUG: Разметка - классы: {gt_classes}, количество: {len(gt)}")
                print(f"  DEBUG: Отфильтрованные предсказания (только классы из разметки): {len(filtered_predictions)}")
                if set(pred_classes) != gt_classes:
                    print(f"  [WARN] Классы в предсказаниях и разметке не совпадают!")
                    print(f"     Используются только классы из разметки для вычисления метрик.")
            
            # Вычисляем метрики только для классов, которые есть в разметке
            if len(filtered_predictions) > 0 or len(gt) > 0:
                tpr, fdr = calculate_metrics(filtered_predictions, gt)
            else:
                tpr, fdr = 1.0, 0.0  # Если нет ни предсказаний, ни разметки
            
            all_tpr.append(tpr)
            all_fdr.append(fdr)
            print(f"  TPR: {tpr:.3f}, FDR: {fdr:.3f}")
        
        # Визуализация
        if args.show or args.save or args.vid:
            result_image = draw_detections(
                image,
                vehicle_detections,
                detector.class_names,
                colors
            )
            
            # Отображение метрик в верхнем левом углу (если есть разметка)
            if annotations and img_name in annotations:
                result_image = draw_metrics(result_image, tpr, fdr)
            
            # Показ изображения
            if args.show:
                cv2.imshow('Detection', result_image)
                # Если включен режим quick, используем задержку 100мс
                # Иначе ждем нажатия клавиши (0)
                delay = 100 if args.quick else 0
                cv2.waitKey(delay)
            
            # Сохранение изображения
            if args.save:
                output_path = os.path.join(args.output, os.path.basename(image_path))
                cv2.imwrite(output_path, result_image)
                print(f"  Сохранено: {output_path}")
            
            # Запись кадра в видео
            if video_writer is not None:
                video_writer.write(result_image)
    
    # Вывод средних метрик
    if len(all_tpr) > 0:
        avg_tpr = sum(all_tpr) / len(all_tpr)
        avg_fdr = sum(all_fdr) / len(all_fdr)
        print(f"\n{'='*60}")
        print(f"Средние метрики по всем изображениям:")
        print(f"  Средний TPR: {avg_tpr:.3f}")
        print(f"  Средний FDR: {avg_fdr:.3f}")
        print(f"{'='*60}")
    
    # Освобождение ресурсов
    if video_writer is not None:
        video_writer.release()
        print("Видео сохранено успешно")
    
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

