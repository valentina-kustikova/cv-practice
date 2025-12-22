import os

class AnnotationLoader:
    def __init__(self, annotation_file):
        self.ground_truth = {}
        self._parse_file(annotation_file)

    def _parse_file(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл разметки не найден: {path}")

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                
                # Формат: FrameID Class X1 Y1 X2 Y2
                frame_id = int(parts[0])
                class_name = parts[1]
                x1, y1, x2, y2 = map(int, parts[2:6])

                if frame_id not in self.ground_truth:
                    self.ground_truth[frame_id] = []

                self.ground_truth[frame_id].append({
                    'class': class_name,
                    'box': (x1, y1, x2, y2)
                })

    def get_boxes(self, frame_id):
        return self.ground_truth.get(frame_id, [])