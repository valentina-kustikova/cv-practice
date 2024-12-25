import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import argparse
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from scipy.cluster.vq import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle

def load_images(folder, label, size, test_size):
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []
    size = size if size < len(os.listdir(folder)) else len(os.listdir(folder))
    train_abs_sz = (int)(size * (1 - test_size))
    i = 0
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv.imread(path)
        if img is not None:
            img = cv.resize(img, (128, 128))
            if (i < train_abs_sz):
                images_train.append(img)
                labels_train.append(label)
            else:
                images_test.append(img)
                labels_test.append(label)
        i = i + 1
        if i > size:
            break
    return images_train, labels_train, images_test, labels_test

def get_sift_descriptors(img_list):
    sift = cv.SIFT_create()
    descriptor_list = []

    for img in img_list:
        _, descriptors = sift.detectAndCompute(img, None)
        descriptor_list.append(descriptors)
    
    return descriptor_list

def clustering(model, descriptor_list):
    stacked_descriptors = descriptor_list[0]
    for descriptor in descriptor_list[1:]:
        stacked_descriptors = np.vstack((stacked_descriptors, descriptor))
    stacked_descriptors = np.float32(stacked_descriptors)

    model.fit(stacked_descriptors)

def vector_quantization(model, descriptor_list, number_of_images, n):
    image_features = np.zeros((number_of_images, n), "float32")

    for i in range(number_of_images):
        if descriptor_list is not None:
            predictions = model.predict(descriptor_list[i])
            for pred in predictions:
                image_features[i][pred] += 1

    return image_features

def normalization(img_feature_list):
    stdscaler = StandardScaler().fit(img_feature_list)
    img_feature_list = stdscaler.transform(img_feature_list)

    return img_feature_list

def KNN(train_feature_list, train_class_list, test_feature_list, test_class_list, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_feature_list, train_class_list)

    y_pred = knn.predict(test_feature_list)
    y_prob = knn.predict_proba(test_feature_list)[:, 1]
    accuracy = accuracy_score(test_class_list, y_pred)
    return accuracy, y_pred, y_prob

def xgb(train_feature_list, train_class_list, test_feature_list, test_class_list, k, depth):
    model = GradientBoostingClassifier(n_estimators=k, learning_rate=0.3, max_depth=depth, random_state=42)
    model.fit(train_feature_list, train_class_list)

    y_pred = model.predict(test_feature_list)
    y_prob = model.predict_proba(test_feature_list)[:, 1]
    accuracy = accuracy_score(test_class_list, y_pred)
    return accuracy, y_pred, y_prob

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='coolwarm')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

class ArgumentParser:
    def cli_argument_parser():
        parser = argparse.ArgumentParser(
            description="Image classification using various algorithms and Bag of Words with SIFT and ORB.")

        parser.add_argument('-td', '--train_dir',
                            help='Directory with training images (cats and dogs)',
                            type=str,
                            dest='train_dir')
        parser.add_argument('-nc', '--n_clusters',
                            help='Number of clusters for visual dictionary',
                            type=int,
                            dest='n_clusters',
                            default=100)
        parser.add_argument('-s', '--dataset_size',
                            help='size of test part compared to train',
                            type=float,
                            dest='size',
                            default='400')
        parser.add_argument('-tp', '--test_proportion',
                            help='size of test part compared to train',
                            type=float,
                            dest='test_prop',
                            default='0.2')

        args = parser.parse_args()
        return args

def main(args):

    cats_train, y_cats_train, cats_test, y_cats_test = load_images(os.path.join(args.train_dir, 'cats'), 0, args.size, args.test_prop)
    dogs_train, y_dogs_train, dogs_test, y_dogs_test = load_images(os.path.join(args.train_dir, 'dogs'), 1, args.size, args.test_prop)

    print(len(cats_train))
    print(len(cats_test))
    model = KMeans(n_clusters=args.n_clusters, random_state=42)
    Train_images = cats_train + dogs_train
    Train_labels = y_cats_train + y_dogs_train
    Test_images = cats_test + dogs_test
    Test_labels = y_cats_test + y_dogs_test

    train_image_list, train_class_list = shuffle(Train_images, Train_labels, random_state=42)
    test_image_list, test_class_list = shuffle(Test_images, Test_labels, random_state=42)

    print("Number of train images = ", len(train_image_list))
    print("Number of test images = ", len(test_image_list))

    train_descriptor_list = get_sift_descriptors(train_image_list)
    test_descriptor_list = get_sift_descriptors(test_image_list)

    clustering(model, train_descriptor_list)
    train_feature_list = vector_quantization(model, train_descriptor_list, len(train_image_list), args.n_clusters)
    print(train_feature_list[0])
    train_feature_list = normalization(train_feature_list)

    test_feature_list = vector_quantization(model, test_descriptor_list, len(test_image_list), args.n_clusters)
    test_feature_list = normalization(test_feature_list)

    accuracy, results, prob = KNN(train_feature_list, train_class_list, test_feature_list, test_class_list, 40)

    print("Test Accuracy on KNN classificator:", accuracy)

    accuracy, results, prob = xgb(train_feature_list, train_class_list, test_feature_list, test_class_list, 80, 5)

    print("Test Accuracy on GB classificator:", accuracy)

    plot_confusion_matrix(test_class_list, results, ['Cat', 'Dog'])

if __name__ == "__main__":
    args = ArgumentParser.cli_argument_parser()
    main(args)