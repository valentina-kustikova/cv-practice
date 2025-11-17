from detector_base import BaseDetector, Detection


class SSDMobileNetDetector(BaseDetector):
    def postprocess(self, outputs, image_shape):
        detections = []
        img_height, img_width = image_shape

        # COCO class IDs for vehicle classes
        COCO_VEHICLE_IDS = {3, 6, 8}  # car, bus, truck

        for detection in outputs[0, 0]:
            confidence = float(detection[2])
            class_id = int(detection[1])

            # ✅ Проверяем по оригинальным COCO ID
            if class_id in COCO_VEHICLE_IDS and confidence > self.config.confidence_threshold:
                # Маппинг в локальное имя
                class_name_map = {3: "car", 6: "bus", 8: "truck"}
                class_name = class_name_map[class_id]

                x1 = int(detection[3] * img_width)
                y1 = int(detection[4] * img_height)
                x2 = int(detection[5] * img_width)
                y2 = int(detection[6] * img_height)

                if x2 > x1 and y2 > y1:
                    detections.append(Detection(
                        class_id=class_id,  # или локальный: COCO_TO_LOCAL[class_id]
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    ))

        return detections