import cv2
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_images_from_folder(folder, label, image_size=(256, 256), max_count=100):
    images, labels = [], []
    for cnt, filename in enumerate(os.listdir(folder)):
        if cnt >= max_count:
            break
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            images.append(cv2.resize(img, image_size))
            labels.append(label)
    return images, labels

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help="Path to source train directory", required=True)
    parser.add_argument("-te", "--test", help="Path to source test directory", required=True)
    parser.add_argument("-k", "--clusters", help="Number of clusters", required=True)
    return parser.parse_args()

def visualize_features(images, sift, number):
    for i, img in enumerate(images[5:number+5]):
        sift_keypoints, _ = get_keypoints_and_descriptors(img, sift)
        img_with_sift_keypoints = cv2.drawKeypoints(img, sift_keypoints, None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(img_with_sift_keypoints, cv2.COLOR_BGR2RGB))
        plt.title('Пример изображения')
        plt.axis('off')
        plt.show()

def plot_classification_histogram(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    correct_counts = np.diag(cm)
    total_counts = np.sum(cm, axis=1)
    accuracy_per_class = correct_counts / total_counts
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracy_per_class, color='green')
    plt.xlabel('Классы')
    plt.ylabel('Точность')
    plt.title('Точность определения классов')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracy_per_class):
        plt.text(i, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=12)
    plt.show()

    
def build_histogram(descriptors, kmeans):
    histogram = np.bincount(
        kmeans.predict(descriptors) if descriptors is not None else [],
        minlength=kmeans.n_clusters
    )
    return histogram

def get_keypoints_and_descriptors(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(gray, None)

def extract_sift_features(images, sift):
    print("Извлечение дескрипторов")
    return [
        descriptors
        for img in images
        if (descriptors := get_keypoints_and_descriptors(img, sift)[1]) is not None
    ]

def build_feature_vectors(images, sift, kmeans):
    return np.array([
        build_histogram(
            descriptors := get_keypoints_and_descriptors(img, sift)[1], 
            kmeans
        )
        for img in images
    ])

def train(X_train, y_train, k, random_state=228):
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

def test(X_test, y_test, sift, kmeans, scaler, clf):
    X_test_features = scaler.transform(build_feature_vectors(X_test, sift, kmeans))
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Точность: {accuracy}")
    #print(f"Classification Report: {classification_report(y_test, y_pred)}")

    return y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix ({dataset_name} sample)')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Добавление значений в ячейки матрицы
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    class_names = ['Кошки', 'Собаки']
    k = int(args.clusters)

    print("Загрузка изображений")
    cats_train, y_cats_train = load_images_from_folder(os.path.join(args.train, 'Cat'), 0)
    dogs_train, y_dogs_train = load_images_from_folder(os.path.join(args.train, 'Dog'), 1)

    cats_test, y_cats_test = load_images_from_folder(os.path.join(args.test, 'Cat'), 0)
    dogs_test, y_dogs_test = load_images_from_folder(os.path.join(args.test, 'Dog'), 1)

    X_train, y_train = cats_train + dogs_train, y_cats_train + y_dogs_train
    X_test, y_test = cats_test + dogs_test, y_cats_test + y_dogs_test

    sift, kmeans, scaler, clf = train(X_train, y_train, k)

    y_test_pred = test(X_test, y_test, sift, kmeans, scaler, clf)

    plot_confusion_matrix(y_test, y_test_pred, class_names, dataset_name="Confusion Matrix")

    visualize_features(X_train, sift, 1)
