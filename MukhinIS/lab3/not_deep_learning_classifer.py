import cv2 as cv
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import argparse
import sys
import random


def cli_argument_parser():
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

    def load_images(self):
        images_train, images_test = [], []
        labels_train, labels_test = [], []
        for filename_train in self.train:
            label = 0 if filename_train.split('/')[-2] == 'Cat' else 1
            path = os.path.join(self.data, filename_train)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, (256, 256))
                images_train.append(img)
                labels_train.append(label)
        for filename_test in self.test:
            label = 0 if filename_train.split('/')[-2] == 'Cat' else 1
            path = os.path.join(self.data, filename_test)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv.resize(img, (256, 256))
                images_test.append(img)
                labels_test.append(label)
        return images_train, labels_train, images_test, labels_test


class Model:
    def __init__(self, clusters):
        self.sift = cv.SIFT_create(nfeatures=500)
        self.clusters = clusters

    def extract_features(self, images):
        self.descriptors = []
        for image in images:
            kp, dp = self.sift.detectAndCompute(image, None)
            if dp is not None:
                self.descriptors.append(dp)
        return self.descriptors

    def create_bow(self):
        self.kmeans = KMeans(self.clusters)
        self.kmeans.fit(np.vstack(self.descriptors))

    def bow_features(self, images):
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
                features.append(np.zeros(self.clusters))
        return np.array(features)


def main():
    args = cli_argument_parser()
    data = Data(args.data_path, args.train_size, args.test_size)
    images_train, labels_train, images_test, labels_test = data.load_images()
    model = Model(args.clusters)
    model.extract_features(images_train)
    model.create_bow()
    train_features = model.bow_features(images_train)
    test_features = model.bow_features(images_test)
    grad = GradientBoostingClassifier()
    grad.fit(train_features, labels_train)
    predictions = grad.predict(test_features)
    acc = accuracy_score(labels_test, predictions)
    print(f'Accuracy = {acc}')






if __name__=='__main__':
    sys.exit(main() or 0)
    