import os
import cv2


class AnnotationParser:
    def __init__(self, annotation_file, images_path):
        self.annotation_file = annotation_file
        self.images_path = images_path
        self.annotations = self._parse_annotations()

    def _parse_annotations(self):
        """Парсинг файла разметки в формате: frame class x1 y1 x2 y2"""
        annotations = {}

        if not os.path.exists(self.annotation_file):
            print(f"Annotation file not found: {self.annotation_file}")
            return annotations

        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 6:
                        print(f"Warning: Line {line_num} has invalid format: {line}")
                        continue

                    try:
                        frame_number = int(parts[0])
                        class_name = parts[1].upper()
                        x1 = int(parts[2])
                        y1 = int(parts[3])
                        x2 = int(parts[4])
                        y2 = int(parts[5])

                        # Проверяем валидность координат
                        if x2 <= x1 or y2 <= y1:
                            print(f"Warning: Invalid bbox coordinates in line {line_num}: {line}")
                            continue

                        # Создаем запись для этого кадра
                        if frame_number not in annotations:
                            annotations[frame_number] = []

                        annotations[frame_number].append({
                            'class_name': class_name,
                            'bbox': (x1, y1, x2, y2),  # Уже готовые координаты
                            'confidence': 1.0
                        })

                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line {line_num}: {line} - {e}")
                        continue

        except Exception as e:
            print(f"Error reading annotation file: {e}")

        print(f"Parsed annotations for {len(annotations)} frames")

        # Статистика по классам
        class_stats = {}
        for frame_anns in annotations.values():
            for ann in frame_anns:
                class_name = ann['class_name']
                class_stats[class_name] = class_stats.get(class_name, 0) + 1

        if class_stats:
            print("Class statistics:")
            for class_name, count in class_stats.items():
                print(f"  {class_name}: {count} objects")

        return annotations

    def get_ground_truth_for_frame(self, frame_number, image_width=None, image_height=None):
        """Получение ground truth для конкретного кадра"""
        frame_annotations = self.annotations.get(frame_number, [])

        # Проверяем координаты на валидность относительно размеров изображения
        if image_width and image_height:
            valid_annotations = []
            for ann in frame_annotations:
                x1, y1, x2, y2 = ann['bbox']
                # Проверяем что bbox внутри изображения
                if (0 <= x1 < image_width and 0 <= x2 <= image_width and
                        0 <= y1 < image_height and 0 <= y2 <= image_height and
                        x2 > x1 and y2 > y1):
                    valid_annotations.append(ann)
                else:
                    print(
                        f"Warning: Invalid bbox for frame {frame_number}: {ann['bbox']} (image size: {image_width}x{image_height})")
            return valid_annotations

        return frame_annotations

    def get_total_frames_with_annotations(self):
        """Получение общего количества кадров с разметкой"""
        return len(self.annotations)

    def get_frame_numbers(self):
        """Получение списка номеров кадров с разметкой"""
        return sorted(self.annotations.keys())

    def get_total_objects(self):
        """Получение общего количества объектов в разметке"""
        total = 0
        for frame_anns in self.annotations.values():
            total += len(frame_anns)
        return total