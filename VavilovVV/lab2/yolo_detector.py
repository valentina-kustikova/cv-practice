import numpy as np
from detector_base import BaseDetector, Detection, ModelConfig


class YOLODetector(BaseDetector):

    # **ИСПРАВЛЕНИЕ:** YOLO требует специальной обработки для получения
    # имен выходных слоев. Мы делаем это в __init__.
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        layer_names = self.net.getLayerNames()
        try:
            # Для OpenCV 4+
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        except AttributeError:
            # Для OpenCV 3
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        print(f"[YOLODetector] Output layers: {self.output_layers}")

    # **ИСПРАВЛЕНИЕ:** Переопределяем 'detect', чтобы передать
    # имена выходных слоев в self.net.forward()
    def detect(self, image):
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)  # Передаем output_layers
        detections = self.postprocess(outputs, image.shape[:2])
        return self._apply_nms(detections)

    def postprocess(self, outputs: list, image_shape):
        detections = []
        img_height, img_width = image_shape

        # **ИСПРАВЛЕНИЕ:** 'outputs' для YOLO - это список (list) numpy-массивов,
        # по одному на каждый выходной слой (self.output_layers).
        # Нам нужно итерироваться по каждому из них.
        for layer_output in outputs:

            # Теперь итерируемся по каждой детекции в этом слое
            for detection in layer_output:
                try:
                    object_confidence = detection[4]

                    # Быстрая отбраковка по objectness
                    if object_confidence < 0.1:
                        continue

                    # **ИСПРАВЛЕНИЕ:** Используем срез на основе реального
                    # количества классов из конфига
                    class_probs = detection[5:5 + len(self.config.classes)]
                    class_id = np.argmax(class_probs)
                    class_confidence = class_probs[class_id]

                    total_confidence = object_confidence * class_confidence

                    # **ГЛАВНОЕ ИСПРАВЛЕНИЕ:**
                    # 1. Проверяем, есть ли class_id в НАШЕМ списке целевых ID
                    # 2. Сравниваем с порогом
                    if class_id in self.target_class_ids and total_confidence > self.config.confidence_threshold:

                        # 2. Получаем имя класса из конфига, а не хардкодим "car"
                        class_name = self.config.classes[class_id]

                        center_x = detection[0] * img_width
                        center_y = detection[1] * img_height
                        width = detection[2] * img_width
                        height = detection[3] * img_height

                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = int(x1 + width)
                        y2 = int(y1 + height)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width - 1, x2)
                        y2 = min(img_height - 1, y2)

                        if x2 > x1 and y2 > y1:
                            detections.append(Detection(
                                class_id=class_id,
                                class_name=class_name,  # Используем настоящее имя
                                confidence=float(total_confidence),
                                bbox=(x1, y1, x2, y2)
                            ))

                except Exception as e:
                    print(f"ERROR processing YOLO detection: {e}")
                    continue

        return detections