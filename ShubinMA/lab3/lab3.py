import cv2 as cv
import numpy as np
import os
from sklearn.metrics import accuracy_score
import argparse
import sys
import random


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Usage mode.',
                        required=True,
                        choices=['Load', 'Fit'],
                        type=str,
                        dest='mode')
    parser.add_argument('-d', '--data',
                        help='Path to directory with "Cat" and "Dog" subdirectories.',
                        required=True,
                        type=str,
                        dest='data_path')
    parser.add_argument('-c', '--classifier',
                        help='Classifer to use.',
                        required=True,
                        choices=['RandomForest', 'GradientBoosting', 'SVC'],
                        type=str,
                        dest='classifier')
    parser.add_argument('-dd', '--detectordescriptor',
                        help='Detector/descriptor to use.',
                        required=True,
                        choices=['SIFT', 'ORB'],
                        type=str,
                        dest='descriptor')
    parser.add_argument('-g', '--glossary',
                        help='Clusterizator to use as "glossary".',
                        required=True,
                        choices=['KMeans', 'BisectingKMeans'],
                        type=str,
                        dest='glossary')
    parser.add_argument('-ls', '--loadsave',
                        help='Directory to use to save/load.',
                        required=False,
                        type=str,
                        dest='loadsave_path')
    parser.add_argument('-tr', '--train_size',
                        help='Train dataset size (even number).',
                        type=int,
                        dest='train_size',
                        default=200)
    parser.add_argument('-ts', '--test_size',
                        help='Test dataset size (even number).',
                        type=int,
                        dest='test_size',
                        default=200)
    parser.add_argument('-cl', '--cluster_num',
                        help='Number of clusters.',
                        type=int,
                        dest='cluster_num',
                        default=50)
    parser.add_argument('-rs', '--random_seed',
                        help='Seed to use in random module.',
                        type=int,
                        dest='random_seed',
                        default=0)
    args = parser.parse_args()

    return args


class DescriptorDetector:
    def __init__(self, _descriptor):
        if _descriptor == 'SIFT':
            self.descriptor = cv.SIFT_create(nfeatures=500)
        else:
            self.descriptor = cv.ORB_create(nfeatures=500)

    def DetectAndCompute(self, _image):
        return self.descriptor.detectAndCompute(_image, None)


class BoWHandler:
    def __init__(self, _cluster_num, _glossary):
        self.cluster_num = _cluster_num
        if _glossary == 'KMeans':
            from sklearn.cluster import KMeans
            self.glossary = KMeans(n_clusters=self.cluster_num, n_init=10)
        else:
            from sklearn.cluster import BisectingKMeans
            self.glossary = BisectingKMeans(n_clusters=self.cluster_num, n_init=10)

    def TrainGlossary(self, _descriptor_detector, _images):
        des_list = []
        for image in _images:
            _, des = _descriptor_detector.DetectAndCompute(image)
            if des is not None:
                des_list.append(des)
        self.glossary.fit(np.vstack(des_list))

    def GetFeatures(self, _descriptor_detector, _images):
        features = []
        for image in _images:
            _, des = _descriptor_detector.DetectAndCompute(image)
            if des is not None:
                feat = np.zeros(self.cluster_num)
                cluster_indices = self.glossary.predict(des)
                for i in cluster_indices:
                    feat[i] += 1
                features.append(feat)
            else:
                features.append(np.zeros(self.cluster_num))
        return np.array(features)


class Classifier:
    def __init__(self, _classifier):
        if _classifier == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            self.classifer = RandomForestClassifier()
        elif _classifier == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifer = GradientBoostingClassifier()
        else:
            from sklearn.svm import SVC
            self.classifer = SVC(kernel='rbf', probability=True, gamma=0.01)

    def Train(self, _features, _labels):
        self.classifer.fit(_features, _labels)

    def Predict(self, _features):
        return self.classifer.predict(_features)


class DataManager:
    def __init__(self, _data_path, _train_size, _test_size):
        self.data = _data_path
        self.train_size = _train_size
        self.test_size = _test_size
        folder_cat = [(os.path.abspath(f'{self.data}/Cat/{i}'), 0) for i in os.listdir(self.data + '/Cat')]
        random.shuffle(folder_cat)
        folder_dog = [(os.path.abspath(f'{self.data}/Dog/{i}'), 1) for i in os.listdir(self.data + '/Dog')]
        random.shuffle(folder_dog)
        self.train = folder_cat[0:(self.train_size//2)] + folder_dog[0:(self.train_size//2)]
        random.shuffle(self.train)
        self.test = folder_cat[(self.train_size//2):(self.train_size//2) + (self.test_size//2)] + folder_dog[(self.train_size//2):(self.train_size//2) + (self.test_size//2)]
        random.shuffle(self.test)

    def LoadData(self):
        images_train, images_test = [], []
        labels_train, labels_test = [], []
        for item_train in self.train:
            path = item_train[0]
            label = item_train[1]
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv.resize(image, (256, 256))
                images_train.append(image)
                labels_train.append(label)
        for item_test in self.test:
            path = item_test[0]
            label = item_test[1]
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv.resize(image, (256, 256))
                images_test.append(image)
                labels_test.append(label)
        return images_train, images_test, labels_train, labels_test


def main():
    args = cli_argument_parser()

    random.seed(args.random_seed)

    if (args.mode == 'Load'):
        if (args.loadsave_path == None):
            raise ValueError("No path to load fit classes specified.")
    if (args.loadsave_path != None):
        import pickle

    data_manager = DataManager(args.data_path, args.train_size, args.test_size)
    images_train, images_test, labels_train, labels_test = data_manager.LoadData()

    decriptor_detector = DescriptorDetector(args.descriptor)

    if (args.mode == 'Load'):
        with open(os.path.abspath(f'{args.loadsave_path}/bow_handler.pickle'), 'rb') as f:
            bow_handler = pickle.load(f)
    else:
        bow_handler = BoWHandler(args.cluster_num, args.glossary)
        bow_handler.TrainGlossary(decriptor_detector, images_train)
    features_train = bow_handler.GetFeatures(decriptor_detector, images_train)
    features_test = bow_handler.GetFeatures(decriptor_detector, images_test)

    if (args.mode == 'Load'):
        with open(os.path.abspath(f'{args.loadsave_path}/classifier.pickle'), 'rb') as f:
            classifier = pickle.load(f)
    else:
        classifier = Classifier(args.classifier)
        classifier.Train(features_train, labels_train)
    predictions_train = classifier.Predict(features_train)
    predictions_test = classifier.Predict(features_test)

    accuracy_train = accuracy_score(labels_train, predictions_train)
    accuracy_test = accuracy_score(labels_test, predictions_test)

    print(f'Accuracy on train data = {accuracy_train*100}')
    print(f'Accuracy on test data = {accuracy_test*100}')

    if (args.mode != 'Load' and args.loadsave_path != None):
        with open(os.path.abspath(f'{args.loadsave_path}/bow_handler.pickle'), 'wb') as f:
            pickle.dump(bow_handler, f)
        with open(os.path.abspath(f'{args.loadsave_path}/classifier.pickle'), 'wb') as f:
            pickle.dump(classifier, f)


if __name__=='__main__':
    sys.exit(main() or 0)
