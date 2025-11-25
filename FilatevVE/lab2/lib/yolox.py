import numpy as np
import cv2
from lib.base_detector import BaseDetector

class YoloX(BaseDetector):
    def __init__(self, modelPath, confThreshold=0.35, nmsThreshold=0.5, objThreshold=0.5, backendId=0, targetId=0):
        self.num_classes = 80
        self.input_size_yolox = (640, 640)
        self.strides = [8, 16, 32]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        
        super().__init__(modelPath, backendId, targetId)
        
        self.generateAnchors()

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def input_size(self):
        return self.input_size_yolox

    def preprocess(self, img):
        blob = np.transpose(img, (2, 0, 1))
        return blob[np.newaxis, :, :, :]

    def postprocess(self, outputs):
        if isinstance(outputs, dict):
            dets = list(outputs.values())[0]
        elif isinstance(outputs, (list, tuple)):
            dets = outputs[0]
        else:
            dets = outputs

        while dets.ndim > 2:
            dets = dets[0]

        if self.grids.ndim == 3:
            grids = self.grids[0]  # (num_detections, 2)
        else:
            grids = self.grids
        
        if self.expanded_strides.ndim == 3:
            expanded_strides = self.expanded_strides[0, :, 0]  # (num_detections,)
        elif self.expanded_strides.ndim == 2:
            expanded_strides = self.expanded_strides[:, 0]  # (num_detections,)
        else:
            expanded_strides = self.expanded_strides

        expanded_strides = expanded_strides.reshape(-1, 1)  # (num_detections, 1)
        
        dets[:, :2] = (dets[:, :2] + grids) * expanded_strides
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * expanded_strides

        boxes = dets[:, :4]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        scores = dets[:, 4:5] * dets[:, 5:]
        max_scores = np.amax(scores, axis=1)
        max_scores_idx = np.argmax(scores, axis=1)

        keep = cv2.dnn.NMSBoxesBatched(boxes_xyxy.tolist(), max_scores.tolist(), max_scores_idx.tolist(), self.confThreshold, self.nmsThreshold)

        candidates = np.concatenate([boxes_xyxy, max_scores[:, None], max_scores_idx[:, None]], axis=1)
        if len(keep) == 0:
            return np.array([])
        return candidates[keep]

    def generateAnchors(self):
        self.grids = []
        self.expanded_strides = []
        hsizes = [self.input_size_yolox[0] // stride for stride in self.strides]
        wsizes = [self.input_size_yolox[1] // stride for stride in self.strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            self.grids.append(grid)
            shape = grid.shape[:2]
            self.expanded_strides.append(np.full((*shape, 1), stride))

        self.grids = np.concatenate(self.grids, 1)
        self.expanded_strides = np.concatenate(self.expanded_strides, 1)
