import sys
import argparse
import cv2
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any


class CommandLineConfig:
    """Класс для обработки аргументов командной строки"""
    
    @staticmethod
    def get_args() -> argparse.Namespace:
        """Получение и обработка аргументов командной строки"""
        cmd_parser = argparse.ArgumentParser(description="YOLO модель для распознавания объектов")
        
        # Обязательные параметры
        required = cmd_parser.add_argument_group('обязательные аргументы')
        required.add_argument("-i", "--image", required=True, 
                              help="Путь к исходному изображению (.jpg)")
        required.add_argument("-m", "--model", required=True, 
                              help="Путь к файлу весов модели")
        required.add_argument("-conf", "--configuration", required=True, 
                              help="Путь к файлу конфигурации")
        required.add_argument("-l", "--labels", required=True, 
                              help="Путь к файлу с названиями классов")
        
        # Опциональные параметры
        cmd_parser.add_argument("-t", "--threshold", type=float, default=0.5,
                             help="Порог вероятности обнаружения (по умолчанию: 0.5)")
        cmd_parser.add_argument("-iou", "--iou-threshold", type=float, default=0.4,
                             help="Порог IoU для подавления дубликатов (по умолчанию: 0.4)")
        cmd_parser.add_argument("-r", "--result", help="Путь для сохранения результата")
        cmd_parser.add_argument("-d", "--display", action="store_true", 
                             help="Отобразить результат в окне")
        
        return cmd_parser.parse_args()


class YoloDetection:
    """Класс для работы с моделью YOLO"""
    
    def __init__(self, config: argparse.Namespace):
        """Инициализация с параметрами из командной строки"""
        self.config = config
        self.threshold = config.threshold
        self.iou_threshold = config.iou_threshold
        self.image_path = config.image
        self.output_path = config.result
        self.display_result = config.display
        
        # Загрузка нейросети и классов
        self.neural_network = None
        self.output_layer_names = None
        self.class_names = []
        self.class_colors = None
        
    def load_resources(self) -> bool:
        """Загрузка всех необходимых ресурсов"""
        try:
            # Загрузка модели
            self.neural_network = cv2.dnn.readNet(self.config.model, self.config.configuration)
            
            # Получение выходных слоев
            all_layers = self.neural_network.getLayerNames()
            unconnected = self.neural_network.getUnconnectedOutLayers()
            self.output_layer_names = [all_layers[i - 1] for i in unconnected]
            
            # Загрузка имен классов
            self.class_names = self._read_class_names(self.config.labels)
            
            # Генерация цветов для визуализации
            self.class_colors = self._generate_colors(len(self.class_names))
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки ресурсов: {e}")
            return False
    
    def _read_class_names(self, class_file: str) -> List[str]:
        """Чтение имён классов из файла"""
        try:
            with open(class_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Ошибка при чтении файла классов: {e}")
            return []
    
    def _generate_colors(self, num_classes: int) -> np.ndarray:
        """Генерация уникальных цветов для классов"""
        return np.random.uniform(0, 255, size=(num_classes, 3))
    
    def process(self) -> int:
        """Основной метод обработки изображения"""
        # Проверка формата файла
        if not Path(self.image_path).suffix.lower() in ['.jpg', '.jpeg']:
            print("Ошибка: поддерживаются только файлы .jpg и .jpeg")
            return 1
        
        # Загрузка изображения
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Ошибка: не удалось загрузить изображение {self.image_path}")
            return 1
        
        # Распознавание объектов
        detections = self.detect_objects(img)
        if detections is None:
            print("Ошибка при распознавании объектов")
            return 1
        
        # Визуализация результатов
        annotated_img, objects_found = self.visualize_detections(img, detections)
        
        # Вывод результатов
        self.print_detection_summary(objects_found)
        self.handle_output(annotated_img)
        
        return 0
    
    def detect_objects(self, image: np.ndarray) -> Optional[Dict]:
        """Поиск объектов на изображении через нейросеть"""
        try:
            height, width = image.shape[:2]
            
            # Подготовка изображения для нейросети
            input_blob = cv2.dnn.blobFromImage(
                image, 1/255.0, (416, 416), 
                swapRB=True, crop=False
            )
            
            # Передача данных в нейросеть
            self.neural_network.setInput(input_blob)
            outputs = self.neural_network.forward(self.output_layer_names)
            
            # Обработка результатов
            boxes = []
            confidences = []
            class_ids = []
            
            # Обработка каждого выхода сети
            for layer_output in outputs:
                for detection in layer_output:
                    # Извлечение вероятностей классов
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = float(scores[class_id])
                    
                    # Фильтрация по порогу уверенности
                    if confidence > self.threshold:
                        # Пересчет координат из относительных в абсолютные
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Расчет верхнего левого угла
                        x = max(0, int(center_x - w / 2))
                        y = max(0, int(center_y - h / 2))
                        
                        boxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(class_id)
            
            # Применение NMS для устранения дублирующихся рамок
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, 
                self.threshold, self.iou_threshold
            )
            
            return {
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids,
                'indices': indices
            }
        except Exception as e:
            print(f"Ошибка при распознавании объектов: {e}")
            return None
    
    def visualize_detections(self, image: np.ndarray, detections: Dict) -> Tuple[np.ndarray, Counter]:
        """Отрисовка результатов распознавания на изображении"""
        result_image = image.copy()
        detected_objects = Counter()
        
        boxes = detections['boxes']
        confidences = detections['confidences']
        class_ids = detections['class_ids']
        indices = detections['indices']
        
        # Проход по всем обнаруженным объектам
        for i in indices:
            # Извлечение данных объекта
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            class_name = self.class_names[class_id]
            color = tuple(map(int, self.class_colors[class_id]))
            
            # Отрисовка рамки
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Отрисовка метки с именем класса и вероятностью
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                result_image, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Подсчет объектов
            detected_objects[class_name] += 1
            
        return result_image, detected_objects
    
    def print_detection_summary(self, objects: Counter) -> None:
        """Вывод статистики по обнаруженным объектам"""
        print("Результаты распознавания:")
        if not objects:
            print("  Объекты не обнаружены")
        else:
            for obj_name, count in objects.items():
                print(f"  {obj_name}: {count}")
    
    def handle_output(self, image: np.ndarray) -> None:
        """Обработка вывода результата (отображение)"""
        
        # Отображение результата в окне
        if self.display_result:
            try:
                # Убедимся, что изображение имеет правильный формат для отображения
                if image.size == 0 or image is None:
                    print("Предупреждение: Изображение пустое или повреждено")
                    return
                
                # Создаем окно с определенными параметрами для лучшего отображения
                window_name = "Результат распознавания объектов"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                # Изменяем размер окна для лучшей видимости
                img_height, img_width = image.shape[:2]
                # Ограничиваем максимальный размер окна
                max_width, max_height = 1200, 800
                scale = min(max_width/img_width, max_height/img_height, 1.0)
                cv2.resizeWindow(window_name, int(img_width*scale), int(img_height*scale))
                
                # Отображаем изображение
                cv2.imshow(window_name, image)
                print("Нажмите любую клавишу для закрытия окна с результатом")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Ошибка при отображении окна: {e}")


def main() -> int:
    """Основная функция программы"""
    try:
        # Получение настроек из командной строки
        args = CommandLineConfig.get_args()
        
        # Инициализация детектора
        detector = YoloDetection(args)
        
        # Загрузка ресурсов
        if not detector.load_resources():
            return 1
        
        # Запуск обработки
        return detector.process()
        
    except Exception as e:
        print(f"Критическая ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())