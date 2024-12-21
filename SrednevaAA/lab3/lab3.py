import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--test_dir',
                        help='Directory with test images',
                        type=str,
                        dest='test_dir'),
    parser.add_argument('-tr', '--train_dir',
                        help='Directory with train images',
                        type=str,
                        dest='train_dir')
    parser.add_argument('-d', '--descriptor',
                        help='Descriptor',
                        type=str, 
                        dest='descriptor',
                        choices=['sift', 'orb'],
                        default='sift')
    parser.add_argument('-n', '--n_components',
                        help='n_components',
                        type=int,
                        dest='n_components',
                        default=100)
    parser.add_argument('-cl', '--classifier',
                        help='Classifier',
                        type=str, 
                        dest='classifier',
                        choices=['svc', 'rf', 'knn', 'gb', 'lr'],
                        default='lr')
    
    args = parser.parse_args()

    return args

def load_img(data_dir, label):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        img = cv.imread(os.path.join(data_dir, filename))
        if img is not None:
            img = cv.resize(img, (256, 256))
            images.append(img)
            labels.append(label)
    return images, labels

def gray_img(images):
    gray_images = []
    for img in images:
        gray_images.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

    return gray_images

def extract_features(gray_images, descriptor):
    if descriptor == 'sift':
        descr = cv.SIFT_create()
    elif descriptor == 'orb':
        descr = cv.ORB_create()
    else:
        raise ValueError('Error with descriptor')
        
    features = []
    for img in gray_images:
        keypoints, desc = descr.detectAndCompute(img, None)
        features.append(desc)

    return features, descr

def create_visual_dictionary(features, n_components):
    features = np.vstack(features)
    kmeans = KMeans(n_clusters=n_components, random_state=42)
    kmeans.fit(features)
    return kmeans

def build_histogram(desc, kmeans, n_components):
    histogram = np.zeros(n_components)
    if desc is not None:
        predictions = kmeans.predict(desc)
        for pred in predictions:
            histogram[pred] += 1
    return histogram

def build_feature_vectors(gray_images, descr, n_components, kmeans):
    features = []
    for img in gray_images:
        _, desc = descr.detectAndCompute(img, None)
        histogram = build_histogram(desc, kmeans, n_components)
        features.append(histogram)
    return np.array(features)

def train_classifier(arg_classifier, feature_vectors, labels):
    if arg_classifier == 'svc':
        classifier = SVC()
    elif arg_classifier == 'rf':
        classifier = RandomForestClassifier(n_estimators=50)
    elif arg_classifier == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif arg_classifier == 'gb':
        classifier = xgb.XGBClassifier()
    elif arg_classifier == 'lr':
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', C=10.0)
    classifier.fit(feature_vectors, labels)
    return classifier

def evaluate_classifier(classifier, test_feature_vectors, y_test):
    predictions = classifier.predict(test_feature_vectors)
    accuracy = np.mean(predictions == y_test)
    return accuracy

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, descr, n_components, arg_classifier): 
    gray_X_train = gray_img(X_train)
    gray_X_test = gray_img(X_test)    
    train_features, desc = extract_features(gray_X_train, descr)
    
    kmeans = create_visual_dictionary(train_features, n_components)
    
    train_feature_vectors = build_feature_vectors(gray_X_train, desc, n_components, kmeans)
    test_feature_vectors = build_feature_vectors(gray_X_test, desc, n_components, kmeans)
    
    classifier = train_classifier(arg_classifier, train_feature_vectors, y_train)
    
    accuracy = evaluate_classifier(classifier, test_feature_vectors, y_test)
    
    return accuracy

def visualize_features(images, detector):
    for i, img in enumerate(images[:5]):
        keypoints, _ = detector.detectAndCompute(img, None)
        img_with_keypoints = cv.drawKeypoints(img, keypoints, None,
                                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        plt.imshow(cv.cvtColor(img_with_keypoints, cv.COLOR_BGR2RGB))
        plt.title('Keypoints')
        plt.axis('off')
        plt.show()
        
        
def main():
    args = cli_argument_parser()
    print("Загрузка изображений")
    cats_train, y_cats_train = load_img(os.path.join(args.train_dir, 'cats'), 0)
    dogs_train, y_dogs_train = load_img(os.path.join(args.train_dir, 'dogs'), 1)
    cats_test, y_cats_test = load_img(os.path.join(args.test_dir, 'cats'), 0)
    dogs_test, y_dogs_test = load_img(os.path.join(args.test_dir, 'dogs'), 1)

    X_train = cats_train + dogs_train
    y_train = y_cats_train + y_dogs_train
    X_test = cats_test + dogs_test
    y_test = y_cats_test + y_dogs_test
   
    print(f"Количество тренировочных изображений: {len(X_train)}")
    print(f"Количество тестовых изображений: {len(X_test)}")
    
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    
    gray = gray_img(X_train);
    
    features, descr = extract_features(gray, args.descriptor)
    
    accuracy = train_and_evaluate_classifier(X_train, y_train, X_test, y_test, args.descriptor, args.n_components, args.classifier)
    
    print(f'Точность на тестовой выборке = {accuracy}')
    
    
    visualize_features(X_train, descr)


if __name__ == '__main__':
    sys.exit(main() or 0)