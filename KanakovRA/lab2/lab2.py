import cv2
import numpy as np
import argparse


# Организация работы с аргументами командной строки
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagePath', type = str)
    parser.add_argument('-v', '--videoPath', type = str)
    parser.add_argument('-c', '--confidence', type = float, default = 0.5)
    parser.add_argument('-n', '--nms', type = float, default = 0.4)
    return parser.parse_args()


# Загрузка предобученной модели YOLOv3
def initializeModel():
    modelCfg, modelWeights, classNames = "yolov3.cfg", "yolov3.weights", "coco.names"

    # Загрузка модели в OpenCV
    net = cv2.dnn.readNet(modelWeights, modelCfg)

    # Получение имен всех выходных слоев нейронной сети
    layerNames = net.getLayerNames()

    # Получение названия несоединенных выходных слоев
    outputLayerNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    # Получение названия всех классов
    classes = []
    with open(classNames, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, outputLayerNames, classes


# Определение глобальных переменных
net, outputLayerNames, classes = initializeModel()


# Функция для анализа выходов нейронной сети
def processNetworkOutput(res: tuple, h: int, w: int,
                         confidence: float) -> list:
    classIds = []
    confidences = []
    boxes = []

    for layersRes in res:
        for detection in layersRes:
            # Первые 4 значения - это координаты центральной точки и ширина с высотой ограничивающего прямоугольника,
            # поэтому отбрасываем их
            scores = detection[5:]

            # Среди оставшихся берем максимум
            classId = np.argmax(scores)
            conf = max(scores)

            # Если confidence меньше выставленного пользователем порога, то не учитываем этот вариант
            if conf > confidence:
                # Получение реальных координат центра прямоугольника и его ширины с высотой
                (centerX, centerY, width, height) = (detection[:4] * np.array([w, h, w, h])).astype(int)

                # Получение координаты левого верхнего угла
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Добавление полученного прямоугольника, а также коэффициент confidence и
                # номер наиболее вероятного класса
                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                classIds.append(classId)

    return classIds, confidences, boxes


# Функция для отрисовки результатов
def drawDetections(image: np.ndarray, boxes: list, confidences: list,
                   classIds: list, boxesIndAfterNMS: np.ndarray) -> np.ndarray:
    stat = {}
    colors = np.random.uniform(0, 255, size = (len(classes), 3))
    for i in range(len(boxes)):
        # Проверка входит ли данный индекс в boxesIndAfterNMS
        if i in boxesIndAfterNMS:
            # Получение расположения прямоугольника
            x, y, w, h = boxes[i]

            # Получение названия класса
            label = str(classes[classIds[i]])

            # Получение случайного цвета для определенного класса
            color = colors[classIds[i]].tolist()

            # Изображение прямоугольника
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Изображение названия класса с коэффициентом confidence
            cv2.putText(image, f"{label} {confidences[i]:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Сохраняем статистику
            if label in stat:
                stat[label] += 1
            else:
                stat[label] = 1
    print(stat)
    return image


def detect(image: np.ndarray, confidence: float, nms: float) -> np.ndarray:
    height, width = image.shape[0], image.shape[1]

    # Подготовка изображения
    # Нормализация значений пикселей в диапазоне от 0 до 1, изменение размера изображения до (416, 416)
    # Изменение RGB на BGR, а также не обрезаем изображение, чтобы не потерять информацию
    blob = cv2.dnn.blobFromImage(image = image, scalefactor = 1 / 255, size = (416, 416),
                                 mean = (0, 0, 0), swapRB = True, crop = False)

    # Установка получившегося изображения как вход в нейронную сеть
    net.setInput(blob)

    # Запуск сети, и получение результата - уверенность сети для каждого выходного слоя
    # Имена выходных слоев нужны, чтобы вернуть только их значения, а не всех слоев сети
    res = net.forward(outputLayerNames)

    # Анализ выходов нейронной сети
    classIds, confidences, boxes = processNetworkOutput(res, height, width, confidence)
    
    # Применение Non-Maximum Suppression (NMS, не-максимальное подавление)
    boxesIndAfterNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms)

    # Отрисовка обнаруженных объектов и вывод статистики в консоль
    image = drawDetections(image, boxes, confidences, classIds, boxesIndAfterNMS)

    return image


def detectVideo(videoPath: str, confidence: float, nms: float) -> None:
    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        raise RuntimeError("Can't open video")
    
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1

        if counter % 10 == 0:
            frame = detect(frame, confidence, nms)

            cv2.imshow("Video frame", frame)

            key = cv2.waitKey(0)
            if key == ord('v'): # Переход к следующему кадру
                continue
            elif key == ord('q'): # Досрочное завершение
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse()

    image = cv2.imread(args.imagePath)
    if image is None:
        print("Can't open image.")

    if args.imagePath:
        image = detect(image, args.confidence, args.nms)
        if image is not None:
            cv2.imshow("Detected Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Use -i flag to provide image path.")
    if args.videoPath:
        detectVideo(args.videoPath, args.confidence, args.nms)
    else:
        print("Use -v flag to provide video path.")


if __name__ == "__main__":
    main()
