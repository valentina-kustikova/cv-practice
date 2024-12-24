import cv2 as cv
import numpy as np
import os
import argparse
import sys
import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BagOfWords:
    def __init__(par, _cluster_num):
        par.cluster_num = _cluster_num
        par.descriptor = cv.SIFT_create(nfeatures=500)
        par.glossary = KMeans(n_clusters=par.cluster_num, n_init=10)

    def Traning(par, _images):
        des_list = []
        for image in _images:
            _, des = par.descriptor.detectAndCompute(image,None)
            if des is not None:
                des_list.append(des)
        par.glossary.fit(np.vstack(des_list))

    def GetWord(par, _images):
        features = []
        for image in _images:
            _, des = par.descriptor.detectAndCompute(image,None)
            if des is not None:
                feat = np.zeros(par.cluster_num)
                cluster_indices = par.glossary.predict(des)
                for i in cluster_indices:
                    feat[i] += 1
                features.append(feat)
            else:
                features.append(np.zeros(par.cluster_num))
        return np.array(features)


class Classifier:
    def __init__(par):
        par.classifier = RandomForestClassifier()

    def Train(par, _features, _labels):
        par.classifier.fit(_features, _labels)

    def Predict(par, _features):
        return par.classifier.predict(_features)


class DataManager:
    def __init__(par, _data_path, _train_size, _test_size):
        par.data = _data_path
        par.train_size = _train_size
        par.test_size = _test_size
        folder_cat = [(os.path.abspath(f'{par.data}/Cat/{i}'), 0) for i in os.listdir(par.data + '/Cat')]
        random.shuffle(folder_cat)
        folder_dog = [(os.path.abspath(f'{par.data}/Dog/{i}'), 1) for i in os.listdir(par.data + '/Dog')]
        random.shuffle(folder_dog)
        par.train = folder_cat[0:(par.train_size//2)] + folder_dog[0:(par.train_size//2)]
        random.shuffle(par.train)
        par.test = folder_cat[(par.train_size//2):(par.train_size//2) + (par.test_size//2)] + folder_dog[(par.train_size//2):(par.train_size//2) + (par.test_size//2)]
        random.shuffle(par.test)

    def LoadData(par):
        images_train, images_test = [], []
        labels_train, labels_test = [], []
        for item_train in par.train:
            path = item_train[0]
            label = item_train[1]
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv.resize(image, (256, 256))
                images_train.append(image)
                labels_train.append(label)
        for item_test in par.test:
            path = item_test[0]
            label = item_test[1]
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv.resize(image, (256, 256))
                images_test.append(image)
                labels_test.append(label)
        return images_train, images_test, labels_train, labels_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data',help='Path to directory with "Cat" and "Dog" subdirectories.',required=True,type=str,dest='data_path')
    parser.add_argument('-tr', '--train_size',help='Train dataset size (even number).',type=int,dest='train_size',default=200)
    parser.add_argument('-ts', '--test_size',help='Test dataset size (even number).',type=int,dest='test_size',default=200)
    parser.add_argument('-cl', '--cluster_num',help='Number of clusters.',type=int,dest='cluster_num',default=50)
    parser.add_argument('-rs', '--random_seed',help='Seed to use in random module.',type=int,dest='random_seed',default=0)
    args = parser.parse_args()

    random.seed(args.random_seed)

    data_manager = DataManager(args.data_path, args.train_size, args.test_size)
    images_train, images_test, labels_train, labels_test = data_manager.LoadData()

    bow_handler = BagOfWords(args.cluster_num)
    bow_handler.Traning(images_train)
    features_train = bow_handler.GetWord(images_train)
    features_test = bow_handler.GetWord(images_test)

    classifier = Classifier()
    classifier.Train(features_train, labels_train)

    predict_test = classifier.Predict(features_test)

    accur_test = accuracy_score(labels_test, predict_test)

    print(f'Точность на тестовой выборке = {accur_test*100}')

if __name__=='__main__':
    sys.exit(main() or 0)
