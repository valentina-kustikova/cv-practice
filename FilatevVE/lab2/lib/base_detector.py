import numpy as np
import cv2
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    
    def __init__(self, modelPath, backendId=0, targetId=0):
        self.modelPath = modelPath
        self.backendId = backendId
        self.targetId = targetId
        self.net = None
        self._load_network()
    
    def _load_network(self):
        self.net = cv2.dnn.readNet(self.modelPath)
        self.net.setPreferableBackend(self.backendId)
        self.net.setPreferableTarget(self.targetId)
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @property
    @abstractmethod
    def input_size(self):
        pass
    
    def setBackendAndTarget(self, backendId, targetId):
        self.backendId = backendId
        self.targetId = targetId
        if self.net is not None:
            self.net.setPreferableBackend(self.backendId)
            self.net.setPreferableTarget(self.targetId)
    
    @abstractmethod
    def preprocess(self, img):
        pass
    
    @abstractmethod
    def postprocess(self, outputs):
        pass
    
    def infer(self, srcimg):
        blob = self.preprocess(srcimg)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        preds = self.postprocess(outs)
        return preds

