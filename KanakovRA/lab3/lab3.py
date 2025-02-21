import cv2
import numpy as np
import argparse
from subprocess import run
import os
import re
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Организация работы с аргументами командной строки
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--trainPath', type = str, default = "cvd/train")
    parser.add_argument('-te', '--testPath', type = str, default = "cvd/test")
    parser.add_argument('-c', '--clusters', type = int, default = 41)
    return parser.parse_args()


# Загрузка изображений
def loadImages(folder: str) -> list:
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img)
    return images


def loadTrainLabels(folder: str) -> np.array:
    labels = []

    for filename in os.listdir(folder):
        # Получаем имя файла без расширения
        name, _ = os.path.splitext(filename)

        # Проверяем, начинается ли имя файла с 'cat' или 'dog'
        if name.startswith('cat'):
            labels.append(0)
        elif name.startswith('dog'):
            labels.append(1)
    return np.array(labels)


def loadTestLabels(folder: str) -> np.array:
    labels = []
 
    for filename in os.listdir(folder):
        # Получаем имя файла без расширения
        name, _ = os.path.splitext(filename)

        # Проверяем, является ли имя файла числом
        if re.match(r'^\d+$', name):
            number = int(name)
            # Если число меньше 101, добавляем 0 в labels
            if number < 101:
                labels.append(0)
            else:
                labels.append(1)
    return np.array(labels)


# Извлечение SIFT-дескрипторов из изображений
def getDescriptors(images: list) -> list:
    sift = cv2.SIFT_create()
    allDescriptors = []
    for image in images:
        # Здесь нам нужны только дескрипторы ключевых точек, сами точки игнорируем
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            allDescriptors.append(descriptors)
    return allDescriptors


# Кластеризация по дескрипторам (создание мешка слов)
def createBow(descriptors: list, clusters: int):
    kmeans = KMeans(n_clusters = clusters, random_state = 42)
    kmeans = kmeans.fit(np.vstack(descriptors))
    return kmeans


# Извлечение признаков на основе мешка слов
def getFeatures(descriptors: list, kmeans, clusters: int):
    features = []
    for desc in descriptors:
        histogram = np.zeros(clusters)
        descLabels = kmeans.predict(desc)
        for label in descLabels:
            histogram[label] += 1
        features.append(histogram)
    return np.array(features)


def main():
    args = parse()

    trainLabels, trainImages = loadTrainLabels(args.trainPath), loadImages(args.trainPath)
    testLabels, testImages = loadTestLabels(args.testPath), loadImages(args.testPath)
    
    trainDesc = getDescriptors(trainImages)
    testDesc = getDescriptors(testImages)

    c = args.clusters
    
    kmeans = createBow(trainDesc, c)

    trainFeatures = getFeatures(trainDesc, kmeans, c)
    testFeatures = getFeatures(testDesc, kmeans, c)

    # Используем метод опорных векторов с линейным ядром для разделения на 2 класса
    model = SVC(kernel = 'rbf', random_state = 42)
    model.fit(trainFeatures, trainLabels)
    pred = model.predict(testFeatures)

    # Вывод в консоль основной статистики
    print(classification_report(testLabels, pred, target_names = ["Cat", "Dog"]))
    
    # Визуализация матрицы рассогласований
    cm = confusion_matrix(testLabels, pred)
    ConfusionMatrixDisplay(cm, display_labels = ["Cat", "Dog"]).plot(cmap = 'Reds')
    plt.show()


if __name__ == '__main__':
    main()
