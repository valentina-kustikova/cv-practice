import cv2 as cv
import numpy as np
import argparse
import sys

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi")
MULTIPLIER = 1./255.

class YOLODetector:
    def __init__(self, weights, config, namesPath):
        self.net = cv.dnn.readNet(weights, config)
        file = open(namesPath)
        self.classes = []
        for line in file.readlines():
            self.classes.append(line.strip())
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.outputLayers = []
        for i in self.net.getUnconnectedOutLayers():
            self.outputLayers.append(self.net.getLayerNames()[i - 1])


    def detectObjects(self, frame, confThreshold=0.5, nmsThreshold=0.5, inputSize=(512, 512)):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, MULTIPLIER, inputSize, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.outputLayers)
        boxes, confidences, classIds = [], [], []
        detectedObjects = {}
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    centerX, centerY = int(detection[0] * w), int(detection[1] * h)
                    width, height = int(detection[2] * w), int(detection[3] * h)
                    x, y = int(centerX - width / 2), int(centerY - height / 2)
                    boxes.append([x, y, width, height])
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    detectedObjects[self.classes[classId]] = detectedObjects.get(self.classes[classId], 0) + 1
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices.flatten():
            x, y, width, height = boxes[i]
            color = self.colors[classIds[i]]
            label = f"{self.classes[classIds[i]]}: {confidences[i]:.3f}"
            cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
        return frame, detectedObjects

    def showStatistic(self, detectedObjects):
        print("Statictic for frame: ")
        for i in detectedObjects.items():     
            print("object: ", i[0], ", number: ", i[1])
        print()

    def processFrameAndShowStat(self, frame, confThreshold, nmsThreshold, inputSize):
        outputFrame, detectedObjects = self.detectObjects(frame, confThreshold, nmsThreshold, inputSize)
        self.showStatistic(detectedObjects)
        return outputFrame

def processVideoFrame(inputPath, yolo, confThreshold, nmsThreshold, inputSize):
    captureFrame = cv.VideoCapture(inputPath)
    while True:
        isSuccess, frame = captureFrame.read()
        if not isSuccess:
            break
        outputFrame = yolo.processFrameAndShowStat(frame, confThreshold, nmsThreshold, inputSize)
        cv.imshow("", frame)
        if cv.waitKey(1) == ord("q"):
            break
    captureFrame.release()

def processImage(inputPath, yolo, confThreshold, nmsThreshold, inputSize):
    image = cv.imread(inputPath)
    if image is None:
        print("Error: Could not load the image.")
        return
    outputImage = yolo.processFrameAndShowStat(image, confThreshold, nmsThreshold, inputSize)
    cv.imshow("", outputImage)
    while True:
        if cv.waitKey(0) == ord("q"):
            break

def processInput(inputPath, yolo, confThreshold, nmsThreshold, inputSize):
    if inputPath.endswith(SUPPORTED_VIDEO_EXTENSIONS):
        processVideoFrame(inputPath, yolo, confThreshold, nmsThreshold, inputSize)
    else:
        processImage(inputPath, yolo, confThreshold, nmsThreshold, inputSize)
    cv.destroyAllWindows()

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True)
    parser.add_argument('-w', required=True)
    parser.add_argument('-cfg', required=True)
    parser.add_argument('-names', required=True)
    parser.add_argument('-confThreshold', type=float, default=0.5)
    parser.add_argument('-nmsThreshold', type=float, default=0.5)
    parser.add_argument('-inputSize', type=int, nargs=2, default=[512, 512])
    return parser.parse_args()

def parseAndShow():
    args = parseArguments()
    yolo = YOLODetector(args.w, args.cfg, args.names)
    processInput(args.i, yolo, args.confThreshold, args.nmsThreshold, tuple(args.inputSize))

def main():
    parseAndShow()

if __name__ == '__main__':
    sys.exit(main() or 0)