import numpy as np
import cv2
from lib.base_detector import BaseDetector

try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False


class SSD(BaseDetector):
    def __init__(self, modelPath, confThreshold=0.5, nmsThreshold=0.4, backendId=0, targetId=0):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.input_size_ssd = (300, 300)
        self.scale = 1.0 / 127.5
        self.mean = 127.5

        self.modelPath = modelPath
        self.backendId = backendId
        self.targetId = targetId
        self.use_onnx_runtime = False

        try:
            super().__init__(modelPath, backendId, targetId)
        except cv2.error as e:
            error_msg = str(e)
            if ("dynamic" in error_msg.lower() or "Shape" in error_msg) and ONNX_RUNTIME_AVAILABLE:
                self.use_onnx_runtime = True
                self.session = ort.InferenceSession(modelPath, providers=['CPUExecutionProvider'])
                self.net = None
            else:
                raise
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def input_size(self):
        return self.input_size_ssd
    
    def setBackendAndTarget(self, backendId, targetId):
        if not self.use_onnx_runtime:
            super().setBackendAndTarget(backendId, targetId)
    
    def preprocess(self, img):
        if self.use_onnx_runtime:
            resized = cv2.resize(img, self.input_size_ssd, interpolation=cv2.INTER_LINEAR)
            if resized.dtype != np.uint8:
                resized = np.clip(resized, 0, 255).astype(np.uint8)
            blob = resized[np.newaxis, :, :, :]
            return blob.astype(np.uint8)
        else:
            blob = cv2.dnn.blobFromImage(
                img, 
                scalefactor=0.007843,
                size=self.input_size_ssd,
                mean=(127.5, 127.5, 127.5),
                swapRB=False,
                crop=False
            )
            return blob
    
    def postprocess(self, outputs):
        if len(outputs) >= 4:
            boxes = outputs[0]  # (1, 100, 4) - [y1, x1, y2, x2]
            classes = outputs[1]  # (1, 100) - class_id
            scores = outputs[2]  # (1, 100) - confidence
            num_detections = int(outputs[3][0]) if len(outputs) > 3 else 100  # количество валидных детекций
        else:
            print(f"Unexpected number of outputs: {len(outputs)}")
            return np.array([])

        boxes = boxes[0]
        classes = classes[0]
        scores = scores[0]

        results = []
        for i in range(min(num_detections, boxes.shape[0])):
            y1 = float(boxes[i, 0]) * self.input_size_ssd[1]
            x1 = float(boxes[i, 1]) * self.input_size_ssd[0]
            y2 = float(boxes[i, 2]) * self.input_size_ssd[1]
            x2 = float(boxes[i, 3]) * self.input_size_ssd[0]
            
            class_id_raw = int(classes[i])
            confidence = float(scores[i])

            if confidence < self.confThreshold or class_id_raw == 0:
                continue

            class_id = class_id_raw - 1

            if class_id < 0 or class_id >= 80:
                continue

            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
            
            results.append([x1, y1, x2, y2, confidence, class_id])
        
        if len(results) == 0:
            return np.array([])
        
        results = np.array(results, dtype=np.float32)

        boxes = results[:, :4]
        scores = results[:, 4]
        class_ids = results[:, 5].astype(np.int32)

        boxes_wh = boxes.copy()
        boxes_wh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_wh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        
        indices = cv2.dnn.NMSBoxes(
            boxes_wh.tolist(),
            scores.tolist(),
            self.confThreshold,
            self.nmsThreshold
        )
        
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            indices = indices.flatten()
            return results[indices]
        else:
            return np.array([])
    
    def infer(self, srcimg):
        if self.use_onnx_runtime:
            blob = self.preprocess(srcimg)
            input_name = self.session.get_inputs()[0].name

            output_names = [output.name for output in self.session.get_outputs()]
            outputs = self.session.run(output_names, {input_name: blob})
            
            preds = self.postprocess(outputs)
        else:
            blob = self.preprocess(srcimg)
            self.net.setInput(blob)
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            preds = self.postprocess(outs)
        
        return preds

