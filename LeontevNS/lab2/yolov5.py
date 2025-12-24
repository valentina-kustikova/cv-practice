import numpy as np
import cv2
import asyncio
from abc import ABC, abstractmethod
from nanodet import Detector

class YOLOv5(Detector):
    def __init__(self, modelPath, confThreshold=0.35, nmsThreshold=0.5, backendId=0, targetId=0):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.net = cv2.dnn.readNet(modelPath)
        self.net.setPreferableBackend(backendId)
        self.net.setPreferableTarget(targetId)
        
    @property
    def name(self):
        return self.__class__.__name__
    
    def setBackendAndTarget(self, backendId, targetId):
        self.net.setPreferableBackend(backendId)
        self.net.setPreferableTarget(targetId)
    
    async def infer(self, srcimg):
        resized = cv2.resize(srcimg, (640, 640))
        
        blob = cv2.dnn.blobFromImage(
            resized, 
            1.0/255.0,
            (640, 640),
            (0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        predictions = outputs[0] if len(outputs) == 1 else outputs
        
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        results = []
        for pred in predictions:
            confidence = pred[4]
            if confidence > self.confThreshold:
                x_center, y_center, width, height = pred[:4]
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                h_scale = srcimg.shape[0] / 640
                w_scale = srcimg.shape[1] / 640
                
                x1 *= w_scale
                y1 *= h_scale
                x2 *= w_scale
                y2 *= h_scale
                
                class_id = np.argmax(pred[5:])
                
                results.append([x1, y1, x2, y2, confidence, class_id])
        
        if not results:
            return np.array([])
        
        detections = np.array(results)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        boxes_wh = boxes.copy()
        boxes_wh[:, 2] = boxes_wh[:, 2] - boxes_wh[:, 0]
        boxes_wh[:, 3] = boxes_wh[:, 3] - boxes_wh[:, 1]
        
        indices = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores.tolist(), 
                                  self.confThreshold, self.nmsThreshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return detections[indices]
        
        return np.array([])
    
    def _convert_to_standard_format(self, preds):
        return preds if len(preds) > 0 else np.array([])
