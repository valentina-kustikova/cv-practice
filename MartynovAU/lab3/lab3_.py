import cv2 as cv
import numpy as np
import os
from sklearn.metrics import accuracy_score
import argparse
import sys
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data',
                        help='Path to directory with "Cat" and "Dog" subdirectories.',
                        required=True,
                        type=str,
                        dest='data_path')
    parser.add_argument('-tr', '--train_size',
                        help='Train dataset size (even number).',
                        type=int,
                        dest='train_size')
    parser.add_argument('-ts', '--test_size',
                        help='Test dataset size (even number).',
                        type=int,
                        dest='test_size')
    parser.add_argument('-cl', '--cluster_num',
                        help='Number of clusters.',
                        type=int,
                        dest='cluster_num')
    parser.add_argument('-rs', '--random_seed',
                        help='Seed to use in random module.',
                        type=int,
                        dest='random_seed',
                        default=0)
    args = parser.parse_args()

    return args

class DescriptorDetector:
    def __init__(self):
        self.descriptor = cv.SIFT_create()

    def DetectAndCompute(self, image):
        return self.descriptor.detectAndCompute(image, None)


class BagOfWords:
    def __init__(self, cluster_numbers):
        self.cluster_num = cluster_numbers
        self.glossary = KMeans(n_clusters=self.cluster_num, init='k-means++', n_init=3)

    def GetFeatures(self, descriptor_detector, images):
        features = []
        for image in images:
            kpoints, descriptors = descriptor_detector.DetectAndCompute(image)
            if descriptors is not None:
                feat = np.zeros(self.cluster_num)
                cluster_indices = self.glossary.predict(descriptors)
                for i in cluster_indices:
                    feat[i] += 1
                features.append(feat)
            else:
                features.append(np.zeros(self.cluster_num))
        return np.array(features)
    
    def Train(self, descriptor_detector, images):
        des_list = []
        for image in images:
            kpoints, descriptors = descriptor_detector.DetectAndCompute(image)
            if descriptors is not None:
                des_list.append(descriptors)
        self.glossary.fit(np.vstack(des_list))

class Classifier:
    def __init__(self):
        self.classifer = RandomForestClassifier()

    def Train(self, _features, labels):
        self.classifer.fit(_features, labels)

    def Predict(self, features):
        return self.classifer.predict(features)

class DataManager:
    def __init__(self, data_path_, train_size_, test_size_):

        if not os.path.exists(data_path_):
            raise FileNotFoundError('file not found - unexist path')
        
        self.data = data_path_
        self.train_size = train_size_
        self.test_size = test_size_
        self.half_test_size = test_size_//2
        self.half_train_size = train_size_//2

        folder_cat = [(os.path.abspath(f'{self.data}/Cat/{i}'), 0) for i in os.listdir(self.data + '/Cat')]
        random.shuffle(folder_cat)

        folder_dog = [(os.path.abspath(f'{self.data}/Dog/{i}'), 1) for i in os.listdir(self.data + '/Dog')]
        random.shuffle(folder_dog)

        self.train = folder_cat[0:(self.half_train_size)] + folder_dog[0:(self.half_train_size)]
        random.shuffle(self.train)

        self.test = folder_cat[(self.half_train_size):(self.half_train_size) + (self.test_size//2)] + folder_dog[(self.half_train_size):(self.half_train_size) + (self.half_train_size)]
        random.shuffle(self.test)

    def LoadData(self):
        images_train, images_test, labels_train, labels_test = [], [], [], []
        for item_train in self.train:
            path = item_train[0]
            label = item_train[1]

            if path is None:
                raise ValueError('Empty path')
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)

            if image is not None:
                image = cv.resize(image, (256, 256))
                images_train.append(image)
                labels_train.append(label)

        for item_test in self.test:
            path = item_test[0]
            label = item_test[1]

            if path is None:
                raise ValueError('Empty path')
            
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv.resize(image, (256, 256))
                images_test.append(image)
                labels_test.append(label)

        if len(set(labels_train)) < 2:
            raise ValueError("Training data contains less than 2 classes!")
        if len(set(labels_test)) < 2:
            raise ValueError("Test data contains less than 2 classes!")    
        return images_train, images_test, labels_train, labels_test

def main():
    args = parser()

    random.seed(args.random_seed)

    data_manager = DataManager(args.data_path, args.train_size, args.test_size)
    images_train, images_test, labels_train, labels_test = data_manager.LoadData()

    decriptors_detector = DescriptorDetector()

    bow = BagOfWords(args.cluster_num)
    bow.Train(decriptors_detector, images_train)

    features_train = bow.GetFeatures(decriptors_detector, images_train)
    features_test = bow.GetFeatures(decriptors_detector, images_test)

    classifier = Classifier()
    classifier.Train(features_train, labels_train)

    predictions_train = classifier.Predict(features_train)
    predictions_test = classifier.Predict(features_test)

    accuracy_train = accuracy_score(labels_train, predictions_train)
    accuracy_test = accuracy_score(labels_test, predictions_test)

    print(f'Accuracy on train data = {accuracy_train}')
    print(f'Accuracy on test data = {accuracy_test}')

if __name__=='__main__':
    sys.exit(main() or 0)