import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
import os
from subprocess import run
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Организация работы с аргументами командной строки
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--trainPath', type = str)
    parser.add_argument('-te', '--testPath', type = str)
    parser.add_argument('-c', '--clusters', type = int, default = 1000)
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


# Загрузка меток
def loadLabels(folder: str) -> list:
    # Выполнение bash-скрипта для генерации файла с метками
    run(['bash', f'{folder}/labels.sh'])
    with open(f'{folder}/labels.txt', 'r', encoding = 'utf-8') as file:
        return list(''.join(char for char in file.read() if char != '\n'))


# Извлечение SIFT-дескрипторов из изображений
def getDescriptors(images):
    sift = cv2.SIFT_create()
    allDescriptors = []
    for image in images:
        # Здесь нам нужны только дескрипторы ключевых точек, сами точки игнорируем
        _, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            allDescriptors.append(descriptors)
    return allDescriptors


# Кластеризация по дескрипторам (создание мешка слов)
def createBow(descriptors, clusters):
    kmeans = KMeans(n_clusters = clusters, random_state = 42)
    kmeans = kmeans.fit(np.vstack(descriptors))
    return kmeans


# Извлечение признаков на основе мешка слов для тестовых данных
def getFeatures(descriptors, kmeans, clusters):
    features = []
    for desc in descriptors:
        histogram = np.zeros(clusters)
        descLabels = kmeans.predict(desc)
        for label in descLabels:
            histogram[label] += 1
        features.append(histogram)
    return np.array(features)


def main():
    args = parse_args()

    trainLabels, trainImages = loadLabels(args.trainPath), loadImages(args.trainPath)
    testLabels, testImages = loadLabels(args.testPath), loadImages(args.testPath)

    trainDesc = getDescriptors(trainImages)
    testDesc = getDescriptors(testImages)

    kmeans = createBow(trainDesc, args.clusters)

    trainFeatures = getFeatures(trainDesc, kmeans, args.clusters)
    testFeatures = getFeatures(testDesc, kmeans, args.clusters)

    # Используем метод опорных векторов для разделения на 2 класса
    model = SVC(kernel = 'linear', random_state = 42)
    model.fit(trainFeatures, trainLabels)
    pred = model.predict(testFeatures)

    accuracy = accuracy_score(testLabels, pred)
    print(f"Accuracy is = {accuracy:.2f}")

    print(classification_report(testLabels, pred, target_names = ["Cat", "Dog"]))

    # Визуализация матрицы рассогласований
    cm = confusion_matrix(testLabels, pred)
    ConfusionMatrixDisplay(cm).plot(cmap = 'Reds')
    plt.show()


if __name__ == '__main__':
    main()
