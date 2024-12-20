import cv2
import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_images_from_folder(folder, label, image_size=(256, 256)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels

def extract_sift_features(images, sift):
    print("Извлечение дескрипторов для всех изображений")
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

def build_histogram(descriptors, kmeans):
    histogram = np.zeros(kmeans.n_clusters)
    if descriptors is not None:
        predictions = kmeans.predict(descriptors)
        for pred in predictions:
            histogram[pred] += 1
    return histogram

def build_feature_vectors(images, sift, kmeans):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        histogram = build_histogram(descriptors, kmeans)
        features.append(histogram)
    return np.array(features)

def train(X_train, y_train, k, random_state=42):
    print("Обучение SVM")
    sift = cv2.SIFT_create()
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=random_state)

    train_descriptors = extract_sift_features(X_train, sift)
    all_descriptors = np.vstack([desc for desc in train_descriptors if desc is not None])
    kmeans.fit(all_descriptors)
    X_train_features = build_feature_vectors(X_train, sift, kmeans)
    X_train_features = scaler.fit_transform(X_train_features)
    clf.fit(X_train_features, y_train)

    return sift, kmeans, scaler, clf

def evaluate(X_test, y_test, sift, kmeans, scaler, clf):
    X_test_features = build_feature_vectors(X_test, sift, kmeans)
    X_test_features = scaler.transform(X_test_features)
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


def visualize_features(images, sift, number):
    for i, img in enumerate(images[3:number+3]):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift_keypoints, _ = sift.detectAndCompute(gray, None)
        img_with_sift_keypoints = cv2.drawKeypoints(img, sift_keypoints, None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(img_with_sift_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('SIFT Keypoints')
        plt.axis('off')
        plt.show()

def plot_classification_histogram(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    correct_counts = np.diag(cm)
    total_counts = np.sum(cm, axis=1)
    accuracy_per_class = correct_counts / total_counts
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracy_per_class, color='skyblue')
    plt.xlabel('Классы')
    plt.ylabel('Точность')
    plt.title('Точность определения классов')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracy_per_class):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontsize=12)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="Path to source train directory", required=True)
    parser.add_argument("-te", "--test", help="Path to source test directory", required=True)
    parser.add_argument("-k", "--clusters", help="Number of clusters", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    print("Загрузка изображений")
    k = int(args.clusters)

    cats_train, y_cats_train = load_images_from_folder(os.path.join(args.train, 'cats'), 0)
    dogs_train, y_dogs_train = load_images_from_folder(os.path.join(args.train, 'dogs'), 1)
    cats_test, y_cats_test = load_images_from_folder(os.path.join(args.test, 'cats'), 0)
    dogs_test, y_dogs_test = load_images_from_folder(os.path.join(args.test, 'dogs'), 1)

    X_train = cats_train + dogs_train
    y_train = y_cats_train + y_dogs_train
    X_test = cats_test + dogs_test
    y_test = y_cats_test + y_dogs_test


    sift, kmeans, scaler, clf = train(X_train, y_train, k)
    #y_train_pred = evaluate(X_train, y_train, sift, kmeans, scaler, clf)
    y_test_pred = evaluate(X_test, y_test, sift, kmeans, scaler, clf)

    #plot_sample_predictions(X_test, y_test, y_test_pred, 12)

    class_names = ['Кошки', 'Собаки']
    #plot_classification_histogram(y_train, y_train_pred, class_names)
    plot_classification_histogram(y_test, y_test_pred, class_names)

    # Вызов функции для отображения изображений c точками отмеченными дескриптором
    visualize_features(X_train, sift, 3)







# def plot_sample_predictions(images, y_true, y_pred, n=12):
#     plt.figure(figsize=(12, 6))
#     for i in range(n):
#         plt.subplot(4, 3, i + 1)
#         img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
#         plt.imshow(img)
#         plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
#         plt.axis('off')
#     plt.show()