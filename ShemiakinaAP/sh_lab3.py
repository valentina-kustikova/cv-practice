#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import argparse
import sys
import random
import matplotlib.pyplot as plt

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data',
                        help='Path to directory with "Cat" and "Dog"'
                             'subdirectories',
                        type=str,
                        required=False,
                        dest='data_path')
    parser.add_argument('--train_size',
                        help='Size of train dataset',
                        type=int,
                        required=False,
                        dest='train_size')
    parser.add_argument('--test_size',
                        help='Size of test dataset',
                        type=int,
                        required=False,
                        dest='test_size')
    parser.add_argument('--clusters',
                        help='Number of clusters',
                        type=int,
                        required=False,
                        default=50,
                        dest='clusters')
    parser.add_argument('--num_images',
                        help='Value of images for visualization descriptors',
                        type=int,
                        required=False,
                        dest='num_images')

    args = parser.parse_args()

    return args


class Data:
    def __init__(self, data_path, train_size, test_size):
        self.data = data_path
        self.train_size = train_size
        self.test_size = test_size
        self.folder_cat = [os.path.abspath(f'{self.data}/Cat/{i}') for i in os.listdir(self.data + '/Cat')]
        self.folder_dog = [os.path.abspath(f'{self.data}/Dog/{i}') for i in os.listdir(self.data + '/Dog')]
        self.folder_all = self.folder_dog + self.folder_cat
        random.shuffle(self.folder_all)
        self.train = self.folder_all[0:self.train_size]
        self.test = self.folder_all[self.train_size:self.train_size + self.test_size]

    def LoadImages(self):
        images_train, labels_train, images_test, labels_test = [], [], [], []

        if self.train_size + self.test_size > len(self.folder_all):
            raise ValueError("Train size and test size exceed the total number of images.")
        for filename_train in self.train:
            label = 0 if os.path.basename(os.path.dirname(filename_train)) == 'Cat' else 1
            path = os.path.join(self.data, filename_train)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE) ##
            if img is not None:
                img = cv.resize(img, (256, 256))
                images_train.append(img)
                labels_train.append(label)
        
        for filename_test in self.test:
            label = 0 if os.path.basename(os.path.dirname(filename_test)) == 'Cat' else 1
            path = os.path.join(self.data, filename_test)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, (256, 256))
                images_test.append(img)
                labels_test.append(label)

        # Проверка на наличие двух классов
        if len(set(labels_train)) < 2:
            raise ValueError("Training data contains less than 2 classes!")
        if len(set(labels_test)) < 2:
            raise ValueError("Test data contains less than 2 classes!")

        return images_train, labels_train, images_test, labels_test

class Model:
    def __init__(self, clusters):
        self.sift = cv.SIFT_create(nfeatures=500)
        self.clusters = clusters

    def ExtractFeatures(self, images):#извлечение признаков
        self.descriptors = []
        for image in images:
            kp, dp = self.sift.detectAndCompute(image, None)
            if dp is not None:
                self.descriptors.append(dp)
        return self.descriptors

    def Create_BagOfWords(self):
        self.kmeans = KMeans(self.clusters)
        self.kmeans.fit(np.vstack(self.descriptors))

    def BowFeatures(self, images):
        features = []
        for img in images:
            _, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is not None:
                histogram = np.zeros(self.clusters)
                cluster_indices = self.kmeans.predict(descriptors)
                for idx in cluster_indices:
                    histogram[idx] += 1
                features.append(histogram)
            else:
                features.append(np.zeros(self.clusters))#вектор признаков
        return np.array(features)

    def VisualizeDescriptors(self, images, num_images):
        # Ограничиваем количество изображений для отображения
        num_images = min(num_images, len(images))

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        
        if num_images == 1:
            axes = [axes]
        
        for i in range(num_images):
            img = images[i]
            gray = img
            kp, _ = self.sift.detectAndCompute(gray, None)
            
            # Отображаем ключевые точки на изображении
            img_with_kp = cv.drawKeypoints(img, kp, None)
            
            # Отображаем оригинальное и обработанное изображение с ключевыми точками
            axes[i].imshow(cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB))
            axes[i].axis('off')
            axes[i].set_title(f"Image {i+1}")
        
        plt.show()

def main():
    args = argument_parser()
    data = Data(args.data_path, args.train_size, args.test_size)
    images_train, labels_train, images_test, labels_test = data.LoadImages()
    model = Model(args.clusters)
    model.ExtractFeatures(images_train)
    model.Create_BagOfWords()
    train_features = model.BowFeatures(images_train)
    test_features = model.BowFeatures(images_test)
    grad = GradientBoostingClassifier()

    grad.fit(train_features, labels_train)
    predictions_test = grad.predict(test_features)
    acc_test = accuracy_score(labels_test, predictions_test)
    print(f'Accuracy on test data = {acc_test}')

    model.VisualizeDescriptors(images_train, args.num_images)

if __name__=='__main__':
    sys.exit(main() or 0)




