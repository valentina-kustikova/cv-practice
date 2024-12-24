import cv2
import os
import glob
import argparse
import sys
import numpy as np
from scipy.cluster.vq import *

from sklearn.utils import shuffle

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-d', '--dataset',
                        help='path to the dataset',
                        type=str,
                        dest='dataset_path')
    
    parser.add_argument('-n', '--clusters',
                        help='n clusters',
                        type=int,
                        dest='n_clusters')
    
    parser.add_argument('-cl', '--classifier',
                        help=' classifier',
                        type=str,
                        dest='classifier')
    
    

    args = parser.parse_args()
    return args


def load_train_test_data(path_data):

    
    images_cat = []
    images_dog = []


    for category in ["Cat/", "Dog/"]:
        file_paths = glob.glob(os.path.join(path_data + category, '*'))[:201]
        
        for file in file_paths:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                image = cv2.resize(image, (200, 200))
                
                if (category == "Cat/"):
                    images_cat.append(image)
                else:
                    images_dog.append(image)
    
                    
    
    X_train = images_cat[:100]
    X_train.extend(images_dog[:100])
    
  
    
    X_test = images_cat[100:]
    X_test.extend(images_dog[100:])

    y_train = [0]*100
    y_train.extend([1]*100)
    y_test  = [0]*100
    y_test.extend([1]*100)
    
    
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        
                
def plot_image(image, label):
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.axis('off')
    
    label = 'Cat' if label == 0 else 'Dog'
    
    plt.title(label)
    
    plt.show()
    
def draw_keypoints(image, keypoints):
    
    img_kp = cv2.drawKeypoints(image, keypoints, cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(255, 0, 0))
    
    plt.imshow(img_kp)
    
    plt.axis('off')
    
    plt.show()
    

def extract_descriptors(images):
    
    sift = cv2.SIFT_create()

    all_descriptors = []
    ctr = 0

    for image in images:
        
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        all_descriptors.append(descriptors)
        
        # if ctr < 3:
        #     draw_keypoints(image, keypoints)
        #     ctr+=1
    return all_descriptors, format_descriptors(all_descriptors)  
                

def create_visual_dictionary(descriptors, k):
    
    centroids, _ = kmeans(descriptors, k, 50)

    
    return centroids  # Центры кластеров — визуальные слова


def visualize_predictions(images, labels, predictions, categories):
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        
        ax.imshow(images[i], cmap='gray')
        
        true_label = categories[labels[i]]
        
        pred_label = categories[predictions[i]]
        
        ax.set_title(f"Истинный: {true_label}\nПредсказанный: {pred_label}")
        
        ax.axis('off')
        
    plt.tight_layout()
    
    plt.show()
    
def format_descriptors(descriptor_list):
    
    stacked_descriptors = descriptor_list[0]
    
    for i in descriptor_list[1:]:
        
        stacked_descriptors = np.vstack((stacked_descriptors, i))

    stacked_descriptors = np.float32(stacked_descriptors)
    
    return stacked_descriptors

def make_histograms(descriptors, kmeans):
    
    V = np.zeros((len(descriptors), len(kmeans)), 'float32')

    for img_idx in range(len(descriptors)): # цикл по дексрипторам изображений
        for desc in descriptors[img_idx]:
            words, _ = vq([desc], kmeans)
            for w in words:
                V[img_idx][w-1] += 1
                
    return V



def main():
    
    args = cli_argument_parser()

    
    path_data = args.dataset_path
    
    X_train, y_train, X_test, y_test = load_train_test_data(path_data)
    X_train, y_train = shuffle(X_train, y_train, random_state=21)
    X_test, y_test = shuffle(X_test, y_test, random_state=21)
    
    # plot_image(X_train[102], y_train[102])
    
    train_descriptors, stacked_train_descriptors = extract_descriptors(X_train)
    test_descriptors, _ = extract_descriptors(X_test)

    n_clusters = args.n_clusters
    kmeans = create_visual_dictionary(stacked_train_descriptors, n_clusters)
    
    
    V = make_histograms(train_descriptors, kmeans)
    test_V = make_histograms(test_descriptors, kmeans)
    
    if args.classifier == "knn":

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(V, y_train)
        pred_knn = knn.predict(test_V)
        accuracy_knn = accuracy_score(y_test, pred_knn)
        
        print("Accuracy_knn:", accuracy_knn)
        
        visualize_predictions(X_test[:5], y_test[:5], pred_knn[:5], ['cat', 'dog'])


    if args.classifier == "svc":

        clf = svm.SVC()
        clf.fit(V, y_train)
        pred_svc = clf.predict(test_V)
    
        accuracy_svc = accuracy_score(y_test, pred_svc)

    
        print("Accuracy_svc:", accuracy_svc)
    
    
    
if __name__ == '__main__':
    sys.exit(main() or 0)

